import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

def main(image_folder, history_visit_num, input_dim, geno_file, weights_file):
    print("Processing ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # call the model
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=2,
        drop_path_rate=0.2,
        global_pool=True,
    )

    # load RETFound weights
    os.chdir('/home/jipengzhang/PycharmProjects/foundation_model/RETFound_MAE')


    checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model.eval()

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    torch.manual_seed(2024)
    np.random.seed(2024)

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    #print("Model = %s" % str(model))
    #summary(model, input_size=(1, 3, 224, 224), row_settings=("depth", "ascii_only"))



    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    img_folder = image_folder
    #img_folder = '/home/jipengzhang/PycharmProjects/foundation_model/RETFound_MAE/test_data/Available_Fundus/'
    embeddings = []
    embeddings_index = []
    for img_name in os.listdir(img_folder):
        img_url = os.path.join(img_folder, img_name)
        if os.path.isfile(img_url):
            embeddings_index.append(img_name)
            img = Image.open(img_url)
            img = img.resize((224, 224))
            img = np.array(img) / 255.
            assert img.shape == (224, 224, 3)
            img = img - imagenet_mean
            img = img / imagenet_std
            x = torch.tensor(img)
            x = x.unsqueeze(dim=0)
            x = torch.einsum('nhwc->nchw', x)
            x = x.to(device)

            hook = model.fc_norm.register_forward_hook(get_activation('second_to_last_layer'))
            output = model(x.float())
            second_to_last_layer_output = activation['second_to_last_layer']
            second_to_last_layer_output = torch.squeeze(second_to_last_layer_output.cpu())
            embeddings.append(second_to_last_layer_output.numpy())
            hook.remove()


    embedding = np.array(embeddings)
    embedding_index = np.array(embeddings_index)

    colnames = []
    for i in range(1024):
        colnames.append('ebd'+str(i))
    ebd = pd.DataFrame(embedding, columns=colnames)
    ebd['img_name'] = embedding_index

    ebd['vis'] = ebd['img_name'].str.split('_').str[1].astype(int)
    ebd = ebd.sort_values(by='vis', ascending=True)
    ebd.reset_index(drop=True, inplace=True)
    ebd['time2last'] = (ebd['vis'] - ebd['vis'].min()) / (ebd['vis'].max() - ebd['vis'].min())

    geno = pd.read_csv(geno_file, sep='\t')
    geno = geno.loc[geno.index.repeat(history_visit_num)].reset_index(drop=True)

    df = pd.concat([ebd, geno], axis=1)

    x_test = df.to_numpy()[:,[*range(1027, 1079), *range(1026, 1027),*range(0, 1024)]]
    x_test = np.vstack(x_test).astype(np.float)
    x_test = x_test.reshape(-1, history_visit_num, input_dim)




    test_set  = TestDataset(x_test)

    batch_size = 32
    test_loader  = DataLoader(test_set, shuffle=False)

    print("Predicting ...")
    hidden_dim = 100  # Hidden dimension
    layer_dim = 1     # Number of hidden LSTM layers
    output_dim = 2    # Output dimension (For example, 1 for regression)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim,device)


    # Load the saved state_dict into the model
    model.load_state_dict(torch.load(weights_file,map_location=device))

    # Set the model to evaluation mode if you are not training
    model.eval()

    # Call the test function
    eval_test = test(model, test_loader, device)

    if eval_test['pred'][0] == 1.0:
        print("Our model indicates a high likelihood of this subject progressing to late AMD over the next two years")
    elif eval_test['pred'][0] == 0.0:
        print("Our model indicates a low likelihood of this subject progressing to late AMD over the next two years")



class TestDataset(Dataset):
    def __init__(self, datax):
        self.datax = datax
        #self.datay = datay
    def __len__(self):
        return self.datax.shape[0]
    def __getitem__(self, ind):
        x = self.datax[ind,:,:] # x = self.datax[ind,:(visit_dim-1),:] for max sample size, x = self.datax[ind,2:(visit_dim-1),:] for fixed sample size
        #y = self.datay[ind]
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.device = device  # Add device attribute

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True).to(device)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device).requires_grad_()

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])
        return out


def test(model, test_loader,device):
    model.eval()
    y_preds = []

    with torch.no_grad():
        for data in test_loader:
            # Process each batch in the test loader
            x = data.to(device).float()
            y_pred = model(x)
            y_preds.append(y_pred)

    # Stack predictions along the first dimension (batch dimension)
    y_preds = torch.cat(y_preds, dim=0)

    # Apply softmax to convert to probabilities
    y_preds = torch.softmax(y_preds, dim=1)

    # If the predictions have fewer dimensions or only one class, adjust accordingly
    if y_preds.shape[1] > 1:
        optimal_threshold = 0.24239542  # Comment this line if determining from validation set
        y_preds_binary = (y_preds[:, 1] >= optimal_threshold).float().cpu().numpy().flatten()
    else:
        # Handling case if there's only a single output
        y_preds_binary = (y_preds >= optimal_threshold).float().cpu().numpy().flatten()

    #print(y_preds_binary)
    return {'pred': y_preds_binary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TV-LSTM Model for Fundus Images and Genotypes")
    parser.add_argument("--image_folder", type=str, default='./test_data/Available_Fundus/', help="Path to the folder containing fundus images")
    parser.add_argument("--history_visit_num", type=int, default=4, help="Number of historical visits")
    parser.add_argument("--input_dim", type=int, default=1077, help="Input dimension")
    parser.add_argument("--geno_file", type=str, default='./test_data/test_subject_52SNPs.txt', help="Path to the genotype file")
    parser.add_argument("--weights_file", type=str, default='TVLSTM_g_4v_weights.pth', help="Path to the LSTM weights file")
    args = parser.parse_args()

    main(args.image_folder, args.history_visit_num, args.input_dim, args.geno_file, args.weights_file)
