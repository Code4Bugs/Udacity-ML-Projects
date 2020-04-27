import time
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models


class FlowerRecognizor():
    def __init__(self, base_model='densenet121', hidden_units=512,
                 learning_rate=0.005, use_gpu=False):
        self.base_model = base_model
        self.hidden_units = hidden_units
        self.use_gpu = use_gpu
        if not use_gpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self._create_model(base_model, hidden_units, learning_rate)
        self.criterion = None

        # print(self.model)

    def _create_model(self, base_model, hidden_units, learning_rate=0.005):
        supported_base_models = {
            'vgg13': models.vgg13,
            'vgg13_bn': models.vgg13_bn,
            'vgg16': models.vgg16,
            'vgg16_bn': models.vgg16_bn,
            'vgg19': models.vgg19,
            'vgg19_bn': models.vgg19_bn,
            'densenet121': models.densenet121,
            'densenet169': models.densenet169
        }
        input_features_dict = {
            'vgg13': 25088,
            'vgg13_bn': 25088,
            'vgg16': 25088,
            'vgg16_bn': 25088,
            'vgg19': 25088,
            'vgg19_bn': 25088,
            'densenet121': 1024,
            'densenet169': 1024
        }
        base_model_function = supported_base_models.get(base_model, None)

        if not base_model_function:
            print("Not a valid base_model. Try: {}".format(
                ','.join(supported_base_models.keys())))

        self.model = base_model_function(pretrained=True)
        input_features = input_features_dict[base_model]

        # Freeze weights of feature extractor.
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.base_model = base_model
        self.model.hidden_units = hidden_units
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_features, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.05)),
            ('fc3', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.model.classifier = classifier

        self.optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=learning_rate)

    def _load_checkpoint(self, model_state_dict, optim_state_dict, class_to_idx):
        self.model.load_state_dict(model_state_dict)
        self.model.class_to_idx = class_to_idx
        self.optimizer.load_state_dict(optim_state_dict)

    @staticmethod
    def load_checkpoint(checkpoint_file, use_gpu=False):
        """
        Creates a model from an existing checkpoint files.
        Input:
        - checkpoint_file: filepath to .pth file
        Output:
        - object of FlowerRecognizor with model loaded from checkpoint
        """

        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        base_model = checkpoint.get("base_model", "densenet121")
        hidden_units = int(checkpoint.get("hidden_units", 512))

        fr = FlowerRecognizor(base_model, hidden_units, use_gpu)

        fr._load_checkpoint(checkpoint['model_state_dict'],
                            checkpoint['optim_state_dict'],
                            checkpoint['class_to_idx'])
        return fr

    def predict(self, image_obj, topk):
        tensor_image = torch.from_numpy(image_obj).type(torch.FloatTensor)
        tensor_image = tensor_image.unsqueeze_(0)

        tensor_image.to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor_image)

            probs = torch.exp(outputs)
            top_p, top_class = probs.topk(topk, dim=1)

            top_p = top_p.numpy()[0]
            top_class = top_class.numpy()[0]
        idx_to_class = {val: key for key, val in
                        self.model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]
        return top_p, top_class

    def _save_model(self, filepath, epochs):
        print(f"Saving model..")
        model_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'base_model': self.model.base_model,
            'class_to_idx': self.model.class_to_idx,
            'optim_state_dict': self.optimizer.state_dict(),
            'nr_epochs': epochs,
            'hidden_units': self.model.hidden_units
        }
        torch.save(model_checkpoint, filepath)

    def _validate(self, valid_loader):
        valid_loss = 0
        valid_accuracy = 0
        for images, labels in valid_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logps = self.model(images)
            loss = self.criterion(logps, labels)

            valid_loss += loss.item()

            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += equals.type(torch.FloatTensor).mean()

        return valid_loss/len(valid_loader), valid_accuracy/len(valid_loader)

    def test(self, test_loader):
        with torch.no_grad():
            test_loss, test_accuracy = self._validate(test_loader)

        print(f"Test loss: {test_loss:.3f}.. "
              f"Test accuracy: {100 * test_accuracy:.2f}%..")

    def train(self, save_dir, train_loader, valid_loader, class_to_idx, epochs):

        self.model.to(self.device)

        self.criterion = nn.NLLLoss()
        train_losses, valid_losses = [], []
        model_save_path = save_dir + "/checkpoint.pth"
        self.model.class_to_idx = class_to_idx

        previous_valid_loss = None
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_train_running_loss = 0
            epoch_batches = 0

            print(f"Epoch {epoch+1}/{epochs}..")

            for images, labels in train_loader:
                epoch_batches += 1

                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                logps = self.model(images)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                epoch_train_running_loss += loss.item()
                if epoch_batches % 10 == 0:
                    print(f"  Batch {epoch+1}.{epoch_batches}/{epochs}.. done")

            else:
                with torch.no_grad():
                    self.model.eval()
                    valid_loss, valid_accuracy = self._validate(valid_loader)
                    valid_losses.append(valid_loss)

                    self.model.train()

                # Save model if it was better.
                if not previous_valid_loss or valid_loss < previous_valid_loss:
                    self._save_model(model_save_path, epoch)
                    previous_valid_loss = valid_loss

            train_losses.append(epoch_train_running_loss/epoch_batches)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Duration {time.time() - epoch_start:.1f}s.. "
                  f"Train loss: {epoch_train_running_loss/epoch_batches:.3f}.."
                  f"Validation loss: {valid_loss:.3f}.. "
                  f"Validation accuracy: {valid_accuracy:.3f}..")