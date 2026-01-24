import numpy as np
import sys
from . import utils
import networkx as nx
import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import ast
import sys
from .ge import LINE
from .models import Net


def hicgnn(input, epoch, output):
    filepath = input
    conversions = "[.1, .1, 2]"
    batch_size = 128
    epochs = epoch
    lr = 0.01
    thresh = 1e-8
    out_path = output

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data')

    if not (os.path.exists(out_path)):
        os.makedirs(out_path)
    if not (os.path.exists(data_dir)):
        os.makedirs(data_dir)

    conversions = ast.literal_eval(conversions)

    if len(conversions) == 3:
        conversions = list(
            np.arange(conversions[0], conversions[2], conversions[1]))
    elif len(conversions) == 1:
        conversions = [conversions[0]]
    else:
        raise Exception('Invalid conversion input.')
        sys.exit(2)

    name = os.path.splitext(os.path.basename(filepath))[0]

    adj = np.loadtxt(filepath)

    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)
    matrix_file = os.path.join(data_dir, name + '_matrix.txt')
    np.savetxt(matrix_file, adj, delimiter='\t')

    # Change to script directory to run the R script
    original_dir = os.getcwd()
    os.chdir(script_dir)
    os.system('Rscript normalize.R ' + name + '_matrix')
    os.chdir(original_dir)

    print('Created normalized matrix form of ' + filepath +
          ' as ' + os.path.join(data_dir, name + '_matrix_KR_normed.txt'))
    normed = np.loadtxt(os.path.join(data_dir, name + '_matrix_KR_normed.txt'))

    G = nx.from_numpy_array(adj)

    embed = LINE(G, embedding_size=512, order='second')
    embed.train(batch_size=batch_size, epochs=epochs, verbose=1)
    embeddings = embed.get_embeddings()
    embeddings = list(embeddings.values())
    embeddings = np.asarray(embeddings)

    data = utils.load_input(normed, embeddings)
    embeddings_file = os.path.join(data_dir, name + '_embeddings.txt')
    np.savetxt(embeddings_file, embeddings)
    print('Created embeddings corresponding to ' +
          filepath + ' as ' + embeddings_file)

    tempmodels = []
    tempspear = []
    tempmse = []
    model_list = []

    for conversion in conversions:
        print(f'Training model using conversion value {conversion}.')
        model = Net()
        criterion = MSELoss()
        optimizer = Adam(
            model.parameters(), lr=lr)

        oldloss = 1
        lossdiff = 1

        truth = utils.cont2dist(data.y, .5)

        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out.float(), truth.float())
            # ===================backward====================
            lossdiff = abs(oldloss - loss)
            loss.backward()
            optimizer.step()
            oldloss = loss
            print(f'Loss: {loss}', end='\r')

        idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1)
        dist_truth = truth[idx[0, :], idx[1, :]]
        coords = model.get_model(data.x, data.edge_index)
        out = torch.cdist(coords, coords)
        dist_out = out[idx[0, :], idx[1, :]]
        SpRho = spearmanr(dist_truth, dist_out.detach().numpy())[0]

        tempspear.append(SpRho)
        tempmodels.append(coords)
        tempmse.append(loss)
        model_list.append(model)

    idx = tempspear.index(max(tempspear))
    repmod = tempmodels[idx]
    repspear = tempspear[idx]
    repmse = tempmse[idx]
    repconv = conversions[idx]
    repnet = model_list[idx]

    print(f'Optimal conversion factor: {repconv}')
    print(f'Optimal dSCC: {repspear}')

    with open(f"{out_path}/" + name + '_log.txt', 'w') as f:
        line1 = f'Optimal conversion factor: {repconv}\n'
        line2 = f'Optimal dSCC: {repspear}\n'
        line3 = f'Final MSE loss: {repmse}\n'
        f.writelines([line1, line2, line3])

    torch.save(repnet.state_dict(), f"{out_path}/" + name + '_weights.pt')
    print('Saved trained model corresponding to ' + filepath +
          ' to ' + f"{out_path}/" + name + '_weights.pt')

    utils.WritePDB(repmod*100, f"{out_path}/" + name + '_structure.pdb')
    print('Saved optimal structure corresponding to ' + filepath +
          ' to ' + f"{out_path}/" + name + '_structure.pdb')
