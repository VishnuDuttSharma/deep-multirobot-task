import time
import torch
torch.set_default_dtype(torch.float32)

from constants import *
from environment import get_reward_grid, run_bfs, to_fov_feature

from torch import nn

# Select gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader

class MyDataloader(torch.utils.data.Dataset):
    """
    Function to create a dataloader. Using the standard definition
    """
    def __init__(self, inp_data, out_data):
        """
        inp_data: Input data
        out_data: Output data
        """
        self.input = torch.FloatTensor(inp_data) # data_size x height x width, data_size  x 1 (channel) x heigth, width
        self.output = torch.LongTensor(out_data)

    def __len__(self):
        """
        Length of the input data
        """
        return self.input.shape[0]

    def __getitem__(self, idx):
        """
        Get a spexific element
        """
        return self.input[idx, :], self.output[idx,0]


def get_data(size, is_test=False):
    """
    Function to get dataloader of a specific size

    Parameters
    ----------
        size: Size of the data
        is_test: Indicate if the data is intended to be used for testing

    Returns
    -------
        Dataloader
        grid_arr: array containing the grids
    """
    # Empty array to save feature and actions
    feat_arr = []
    act_arr = []

    # For test data we wuill save grid as well (for plotting)
    if is_test:
        grid_arr = []

    # Generate features (each feature list will already have NUM_ROBOT features)
    for i in range(size//NUM_ROBOT):
        # Create a random reward grid
        grid = get_reward_grid(HEIGHT, WIDTH, REWARD_THRESH)
        # Generate random inital locations
        # intial_pos = STEP*np.array([np.random.randint(0, grid.shape[0]//STEP, NUM_ROBOT), np.random.randint(0, grid.shape[1]//STEP, NUM_ROBOT)])
        intial_pos = np.zeros((NUM_ROBOT, 2), dtype=int)
        rows, cols = np.where(grid > 0)
        indices = list(zip(rows, cols))
        cand_indices = np.random.choice(range(len(indices)), NUM_ROBOT)
        for idx, indc in enumerate(cand_indices):
            intial_pos[idx,:] = np.array(indices[indc])
            grid[intial_pos[idx,0], intial_pos[idx,1]] = 0
        
        
        # Get bets possible paths for each robot
        gt_paths = run_bfs(intial_pos, grid)
        
        ## Debugging
        # print(gt_paths)

        # # Convert the ground truth to features and actions
        # feat, act = to_feature(gt_paths, grid)

        # Convert the ground truth to features and actions with fov
        feat, act = to_fov_feature(gt_paths, grid, fov=FOV)

        # Save features and actions into lists
        feat_arr.append(feat)
        act_arr.append(act)

        # For test data, save grids
        if is_test:
            grid_arr.append(grid)

    # For test data return grid array along with dataloader
    if is_test:
        return MyDataloader(np.array(feat_arr).reshape(-1,4), np.array(act_arr).reshape(-1,1)), grid_arr
    else:
        return MyDataloader(np.array(feat_arr).reshape(-1,4), np.array(act_arr).reshape(-1,1))

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    """
    model: input model
    train_dl: dataloader for training data
    valid_dl: dataloader for validation data
    loss_fn: Loss function
    optimizer: Optimizer
    acc_fn: Accuracy. Doesn't help with training
    """
    start = time.time()

    # send model gpu/cpu
    model.to(device)

    train_loss, valid_loss = [np.inf], [np.inf]

    best_acc = 0.0

    for epoch in range(epochs): #epochs
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode. Switch regularizers off
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.to(device) # input image
                y = y.to(device) # output image
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    # inference
                    outputs = model(x)

                    # find loss
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    model.eval()
                    with torch.no_grad(): # doesn't calculate gradients
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                    model.train()

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size

                if step % 200 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            if phase=='train':
                train_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)

                if valid_loss[-1] < np.array(valid_loss[:-1]).min():
                    model.eval();
                    torch.save(model, 'best_model.pth')
                    model.train();

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss

def acc_metric(predb, yb):
    """
    Avegrage over all pixels, calculate prediction correct and not
    """
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()



# Create a PyTorch network

net = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4)
        ).to(device)

