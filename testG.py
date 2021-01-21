import os
#from custom_data_io import custom_dset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
#import Models as models
import load_data
import cross_val
import util
import argparse
import encoders


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Testing settings
#batch_size = 1
no_cuda = False
# seed = 8
#target_list = "./depth_500/DYGZ_deep_500.csv"
#target_name = 'unknown zone'
ckpt_model = './ckpt/zp_batchsiez=40/model_epoch348.pth'  # todo
#ckpt_path='./ckpt/zp_batchsiez=40/'
#ckpt_model = './ckpt_reasonable/model_epoch22.pth'
cuda = not no_cuda and torch.cuda.is_available()

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=False,#default can be False or Ture#todo
                        help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')
    parser.add_argument('--seed', dest='seed',
                        help='random seed')
    parser.set_defaults(bmname='ZP1',

                        datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=2592,
                        cuda='0',
                        feature_type='default',
                        lr=0.01,
                        clip=2.0,
                        batch_size=1,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=6,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1,
                        seed=4
                        )
    return parser.parse_args()


#target_dataset = custom_dset(txt_path=target_list, nx=227, nz=227, labeled=True)

# target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True,
#                                  drop_last=False)




def load_ckpt(model):
    model.load_state_dict(torch.load(ckpt_model))
    return model


def predict(model):
    model.eval()


    output_file = open("prediction_ZP2.txt", 'w')#todo
    output_file.write('Sample, ' + 'Label, ' + 'Prediction' + '\n')

    labels = []
    scores = []

    #source code
    for batch_idx, data in enumerate(test_loader):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda() #adj need to be modified
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        ypred = torch.softmax(ypred,dim=1)
        scores.append(ypred.cpu().data.numpy())
        print(str(batch_idx) + ', ' + str(labels[batch_idx][0]) + ', ' + str(scores[batch_idx][0,1]) + '\n')
        output_file.write(str(batch_idx) + ', ' + str(labels[batch_idx][0]) + ', ' + str(scores[batch_idx][0,1]) + '\n')
    output_file.close()

    # for batch_idx, data in enumerate(test_loader):
    #     adj = Variable(torch.from_numpy(data['adj']).float(), requires_grad=False).cuda() #adj need to be modified
    #     h0 = Variable(torch.from_numpy(data['feats']).float()).cuda()
    #     labels.append(data['label'])
    #     batch_num_nodes = data['num_nodes']
    #     assign_input = Variable(torch.from_numpy(data['assign_feats']).float(), requires_grad=False).cuda()
    #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
    #     scores.append(ypred.cpu().data.numpy())
    #     print(str(batch_idx) + ', ' + str(labels.numpy()[batch_idx]) + ', ' + str(scores[batch_idx]) + '\n')
    #     output_file.write(str(batch_idx) + ', ' + str(labels.numpy()[batch_idx]) + ', ' + str(scores[batch_idx]) + '\n')
    # output_file.close()

    # for batch_idx, data in enumerate(test_dataset):
    #     adj = Variable(data['adj'], requires_grad=False).cuda() #adj need to be modified
    #     h0 = Variable(data['feats']).cuda()
    #     labels.append(data['label'])
    #     batch_num_nodes = data['num_nodes']
    #     assign_input = Variable(data['assign_feats'], requires_grad=False).cuda()
    #     ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
    #     scores.append(ypred.cpu().data.numpy())
    #     print(str(batch_idx) + ', ' + str(labels.numpy()[batch_idx]) + ', ' + str(scores[batch_idx]) + '\n')
    #     output_file.write(str(batch_idx) + ', ' + str(labels.numpy()[batch_idx]) + ', ' + str(scores[batch_idx]) + '\n')
    # output_file.close()

if __name__ == '__main__':
    # model = models.DANNet(num_classes=2)  # Models.py#todo:
    #model = models.DAN_with_Alex(num_classes=2)
    args = arg_parse()

    Hlist, New_G = load_data.read_graphfile2(args.datadir, args.bmname, max_nodes=args.max_nodes)
    example_node = util.node_dict(New_G)[0]

    input_dim = New_G.graph['feat_dim']

    test_loader, max_num_nodes, input_dim, assign_input_dim = \
        cross_val.prepare_val_data3(Hlist, New_G, args, max_nodes=args.max_nodes)
    #len_target_dataset = len(test_dataset)
    len_target_loader = len(test_loader)
    model = encoders.SoftPoolingGcnEncoder(
        max_num_nodes,
        input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        assign_input_dim=assign_input_dim).cuda()
    correct = 0
    print(model)
    if cuda:
        model.cuda()

    model = load_ckpt(model)
    predict(model)
