#learn heuristic for the game of teeko
import torch
import csv
import ast
import torch.utils
from tqdm import tqdm

def read_csv(file):
    """
    Read the csv file and return the data as a list
    :param file: the file to read
    :return: the data as a list
    """
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        #convert the state of the board to a 25 element list
        for i in range(len(data)):
            tmp = ast.literal_eval(data[i][0])
            #print(tmp)
            #convert to 5x5 tensor
            data[i][0] = torch.zeros(1,5,5)
            data[i][1] = torch.tensor(float(data[i][1]) / 100.0)
            for j in range(5):
                for k in range(5):
                    if tmp[j][k] == 'r':
                        data[i][0][0][j][k] = 1.0
                    elif tmp[j][k] == 'b':
                        data[i][0][0][j][k] = -1.0
                    else:
                        data[i][0][0][j][k] = 0.0
    return data

def build_model():
    """
    Build the neural network model
    :return: the model
    """
    #simple neural network with 2 hidden layers
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(16*5*5, 25),
        torch.nn.ReLU(),
        torch.nn.Linear(25, 1)
    )
    #model = torch.nn.Linear(25,1)
    return model

def train_model(model, data, criterion, optimizer, T, batch_size=1):
    """
    Train the model
    :param model: the model to train
    :param data: the data to train on
    :param criterion: the loss function
    :param optimizer: the optimizer
    :param T: the number of epochs
    :param batch_size: the batch size
    :return: the trained model
    """
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    for t in tqdm(range(T)):
        for x, y in dataloader:
            # Forward pass: Compute predicted y by passing x to the model
            #print(x.shape)
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred.view(-1), y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def save_model(model, file):
    """
    Save the model to a file
    :param model: the model to save
    :param file: the file to save the model to
    :return: None
    """
    torch.save(model, file)

def predict(model, file):
    """
    Predict the value of a state
    :param model: the model to use
    :param file: the file containing the states
    :return: the predicted value
    """
    data = read_csv(file)
    for i in range(min(100, len(data))):
        x = torch.tensor(data[i][0])
        #print(x)
        output = model(x)
        print(output)

def main():
    data = read_csv('test_data.csv')
    model = build_model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    T = 100
    batch_size = 40
    model = train_model(model, data, criterion, optimizer, T, batch_size)
    save_model(model, 'teeko_model.pth')
    #predict(model, 'test_data.csv')

def recover_weights():
    model = torch.load('teeko_model.pth')
    print(model)
    for param in model.parameters():
        print(param)
    print(model.state_dict())
    for key in model.state_dict():
        print(key)
        print(model.state_dict()[key])

if __name__ == '__main__':
    main()
    #print(read_csv('test_data.csv'))
    #print(read_csv('test_data.csv'))
    #predict(torch.load('teeko_model.pth'), 'test_data.csv')
    #recover_weights()