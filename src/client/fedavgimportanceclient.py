import copy
import torch
import inspect
import itertools
import numpy as np

from .baseclient import BaseClient
from src import MetricManager


class FedavgimportanceClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgimportanceClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        self.sampled_training_set = training_set
        self.sampled_train_loader = None

        self.data_importance = None
        self.data_importance_history = []

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def get_data_importance(self):
        """
        Calculates the data importance for each sample in the training set.

        Args:
            model: The PyTorch model used for prediction.

        Returns:
            A list of data importance values (squared L2 norm of prediction error) for each sample.
        """

        data_importance = []
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for data, target in self.train_loader:
                # Move data and target to device (CPU or GPU) if applicable
                data, target = data.to(self.args.device), target.to(self.args.device)
                
                num_classes = 10
                target = torch.nn.functional.one_hot(target, num_classes=num_classes)


                # Get model output before softmax
                output = self.model(data)
                if not isinstance(output, torch.Tensor):
                    output = output[0]  # Assuming logits are the first element in the output
                
                # Calculate difference between logits and true labels
                diff = torch.nn.functional.softmax(output) - target
                

                # Calculate squared L2 norm of the difference for each sample
                sample_importance = torch.linalg.vector_norm(diff, dim=1) ** 2  # L2 norm across dimensions
                data_importance.append(sample_importance.cpu().numpy().tolist())  # Convert to list

                #print(len(data_importance))
                #print(data_importance[0])
        self.data_importance = data_importance
        print(np.mean(list(itertools.chain.from_iterable(data_importance))))
        return data_importance

    def download(self, model):
        self.model = copy.deepcopy(model)

    def upload(self):
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])
    
    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
