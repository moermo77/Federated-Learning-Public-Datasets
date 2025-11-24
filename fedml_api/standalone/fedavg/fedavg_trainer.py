import copy
import logging
import imageio
import numpy as np
import wandb
import matplotlib.pyplot as plt
from fedml_api.standalone.fedavg.client import Client
import torch
import io
from PIL import  Image

class FedAvgTrainer(object):
    def __init__(self, dataset, model, device, args, round = 0, shift = 0):
        self.training_setup_seed(0)
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.model = model
        self.model_local = copy.deepcopy(model)
        self.model.train()
        self.model_local.train()
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)
        self.round = round
        self.shift = shift

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, self.model)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            #模拟中途加入
            if(round_idx<self.round):
                client_indexes = [client_index for client_index in range(client_num_in_total-self.shift)]
            else:
                client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        #logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def train(self,data_dir):
        w_global = self.model.state_dict()
        im_list = []
        for round_idx in range(self.args.comm_round):

            w_locals, loss_locals = [], []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self.client_sampling(round_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)
            #logging.info("client_indexes = " + str(client_indexes))
            for idx, client in enumerate(self.client_list):
                # update dataset
                if(idx<np.shape(client_indexes)[0]):
                    client_idx = client_indexes[idx]
                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])
                    # train on new dataset
                    w, loss = client.train(w_global,round_idx)
                    # self.logger.info("local weights = " + str(w))
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    loss_locals.append(copy.deepcopy(loss))
                    #logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

            # update global weights
            w_global = self.aggregate(w_locals)
            # logging.info("global weights = " + str(w_glob))

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            #logging.info('Rouwnd {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))
            #wandb.log({"local_model-local_test#test_acc":loss_avg})
            if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                logging.info("################Communication round : {}".format(round_idx))
                self.model.load_state_dict(w_global)
                self.local_test_on_all_clients(self.model, round_idx)
                im_list.append(self.gen_image_list(epoch=round_idx))
        #self.create_gif(im_list,data_dir+'/line.gif')
        torch.save(self.model, data_dir+'/model/model_globle.pkl')
        if(self.args.local_fine_tuning):
            client_indexes = self.client_sampling(0, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)
            client_indexes = [len(client_indexes)-1]
            logging.info("client_indexes = " + str(client_indexes))
            logging.info("################Local Fine Tuning")
            test_metrics = {
                'num_samples': [],
                'num_correct': [],
                'precisions': [],
                'recalls': [],
                'losses': []
            }
            for idx, client in enumerate([self.client_list[0]]):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                w, loss = client.fine_tuning(w_global,0.1)
                self.model_local.load_state_dict(w)

                self.model.load_state_dict(w_global)
                test_local_metrics = client.local_test(self.model, True)

                test_metrics['num_samples'] = copy.deepcopy(test_local_metrics['test_total'])
                test_metrics['num_correct'] = copy.deepcopy(test_local_metrics['test_correct'])
                test_metrics['losses'] = copy.deepcopy(test_local_metrics['test_loss'])
                test_acc = test_metrics['num_correct'] / test_metrics['num_samples']
                test_loss = test_metrics['losses'] / test_metrics['num_samples']
                #wandb.log({"global_model-local_test#test_acc": test_acc, "test_loss":test_loss})
                #logging.info({"global_model-local_test#test_acc": test_acc, "test_loss":test_loss})
                logging.info({"global_model-local_test#test_MSE": test_loss})

                test_local_metrics = client.local_test(self.model_local, True)
                test_metrics['num_samples']=copy.deepcopy(test_local_metrics['test_total'])
                test_metrics['num_correct']=copy.deepcopy(test_local_metrics['test_correct'])
                test_metrics['losses']=copy.deepcopy(test_local_metrics['test_loss'])
                test_acc = test_metrics['num_correct'] / test_metrics['num_samples']
                test_loss = test_metrics['losses'] / test_metrics['num_samples']
                #wandb.log({"local_model-local_test#test_acc":test_acc,"test_loss":test_loss})
                #logging.info({"local_model-local_test#test_acc":test_acc,"test_loss":test_loss})
                logging.info({"local_model-local_test#test_MSE":test_loss})
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_global,
                                            self.train_data_local_num_dict[client_idx])

                test_local_metrics = client.local_test(self.model, True)
                test_metrics['num_samples'] = copy.deepcopy(test_local_metrics['test_total'])
                test_metrics['num_correct'] = copy.deepcopy(test_local_metrics['test_correct'])
                test_metrics['losses'] = copy.deepcopy(test_local_metrics['test_loss'])
                test_acc = test_metrics['num_correct'] / test_metrics['num_samples']
                test_loss = test_metrics['losses'] / test_metrics['num_samples']
                #wandb.log({"gloal_model-global_test#test_acc": test_acc, "test_loss": test_loss})
                logging.info({"gloal_model-global_test#test_acc": test_acc, "test_loss": test_loss})

                test_local_metrics = client.local_test(self.model_local, True)
                test_metrics['num_samples']=copy.deepcopy(test_local_metrics['test_total'])
                test_metrics['num_correct']=copy.deepcopy(test_local_metrics['test_correct'])
                test_metrics['losses']=copy.deepcopy(test_local_metrics['test_loss'])
                test_acc = test_metrics['num_correct'] / test_metrics['num_samples']
                test_loss = test_metrics['losses'] / test_metrics['num_samples']
                #wandb.log({"local_model-global_test#test_acc": test_acc, "test_loss": test_loss})
                logging.info({"local_model-global_test#test_acc": test_acc, "test_loss": test_loss})
                torch.save(self.model, data_dir+'/model/model_local_finetune_'+str(idx)+'.pkl')
            torch.save(self.model, data_dir+'/model/model_globle_finetune.pkl')



    def aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def create_gif(self,image_list, gif_name, duration=0.35):
        image_list[0].save(gif_name, save_all=True, append_images=image_list[1:], loop=0, disposal=2)

    def gen_image_list(self,x = np.linspace(-2,5),epoch = 0):
        x = x.astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(1).cuda()
        y_t = self.model.forward(x_t)
        fig=plt.figure("Image",frameon=False)
        canvas = fig.canvas
        x = x_t.detach().cpu().numpy()
        y = y_t.detach().cpu().numpy()
        plt.cla()
        plt.xlim(-2.5,5.5)
        plt.ylim(-3,10.5)
        plt.plot(x,y)
        plt.title('epoch:'+str(epoch))
        plt.plot(x,x**2-3*x+0.1)
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        img = Image.open(buffer)
        return img



    def local_test_on_all_clients(self, model_global, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
        train_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : [],
            'r2': []
        }

        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : [],
            'r2': []
        }

        client = self.client_list[0]
        if round_idx<self.round:
            clinet_num = self.args.client_num_in_total-self.shift
        else:
            clinet_num = self.args.client_num_in_total
        for client_idx in range(clinet_num):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train test
            train_local_metrics = client.local_test(model_global, False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))
            train_metrics['r2'].append(copy.deepcopy(train_local_metrics['test_r2']))

            # test test
            test_local_metrics = client.local_test(model_global, True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            test_metrics['r2'].append(copy.deepcopy(test_local_metrics['test_r2']))

            if self.args.dataset == "stackoverflow_lr":
                train_metrics['precisions'].append(copy.deepcopy(train_local_metrics['test_precision']))
                train_metrics['recalls'].append(copy.deepcopy(train_local_metrics['test_recall']))
                test_metrics['precisions'].append(copy.deepcopy(test_local_metrics['test_precision']))
                test_metrics['recalls'].append(copy.deepcopy(test_local_metrics['test_recall']))

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
        train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])
        train_r2 = sum(train_metrics['r2']) / len(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])
        test_r2 = sum(test_metrics['r2']) / len(test_metrics['num_samples'])
        if self.args.dataset == "stackoverflow_lr":
            stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Pre": train_precision, "round": round_idx})
            wandb.log({"Train/Rec": train_recall, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_precision, "round": round_idx})
            wandb.log({"Test/Rec": test_recall, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)

        else:
            stats = {'training_loss': train_loss, 'train_r2': train_r2}
            #wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, 'train_r2': train_r2, "round": round_idx})
            logging.info(stats)

            stats = {'test_loss': test_loss, 'test_r2': test_r2}
            #wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, 'test_r2': test_r2, "round": round_idx})
            logging.info(stats)

            mse = np.array(test_metrics['losses'])/np.array(test_metrics['num_samples'])
            r2 = np.array(test_metrics['r2'])
            logging.info({"Client MSE: ": mse})
            logging.info({"Client R2: " : r2})
            for i in range(clinet_num):
                wandb.log({"Test/MSE:"+str(i): mse[i], "round": round_idx})

    def training_setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True