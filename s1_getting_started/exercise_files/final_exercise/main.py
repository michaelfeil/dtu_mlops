import argparse
import sys

import torch
import torch.nn as nn
from data import mnist, CustomTensorDataset
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
        
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here        
        model = MyAwesomeModel()
        model = model.to(self.device)
        (X_train, y_train), (X_test, y_test) = mnist()

        train_dataset_normal = CustomTensorDataset(tensors=(X_train, y_train), transform=None)
        trainloader = torch.utils.data.DataLoader(train_dataset_normal, batch_size=16)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        criterion = nn.CrossEntropyLoss() 
        
        epochs = 1
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                # Move to GPU for acceleration
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                
            else:
                ## TODO: Implement the validation pass and print out the validation accuracy
                with torch.no_grad():   
                    # images, labels = next(iter(testloader))
                    images, labels = X_test.to(self.device), y_test.to(self.device)
                    ps = torch.exp(model(images))
                    acc = torch.eq(labels.argmax(1), ps.argmax(1)).sum() / labels.size(0)
                        
                print(f'Accuracy: {acc.item()}, running_loss {running_loss}')  
        print("Training done")
        # TODO save model
        path = './checkpoint.pth'
        torch.save(model.cpu(), path)
        print(f"saved model to {path}")
                
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        (_, _), (X_test, y_test) = mnist()
        ps = model(X_test)
        ps = torch.exp(ps)
        acc = torch.eq(y_test.argmax(1), ps.argmax(1)).sum() / y_test.size(0)
        print(f"evaluted acc is: {acc}")

if __name__ == '__main__':
    
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    