import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np
from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from reader.reader import init_dataset

logger = logging.getLogger(__name__)




def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    ###
    '''
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }
    '''


    model_to_save = model_to_save.state_dict()

    for key in model_to_save.keys():
        if "embeddings.prompt_embeddings.weight" in key:
            if "roberta" in key:
                prompt_emb = model_to_save["encoder.roberta.embeddings.prompt_embeddings.weight"]
            elif "bert" in key:
                prompt_emb = model_to_save["encoder.bert.embeddings.prompt_embeddings.weight"]
    ###


    save_params = {
        "model": prompt_emb,
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }


    try:
        ###
        #torch.save(save_params, filename)
        filename = filename.replace(".pkl","_task_prompt.pkl")
        #print("====")
        #print(filename)
        #print("====")
        #torch.save(prompt_emb, filename)
        torch.save(save_params, filename)
        ###
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list, do_test=False, local_rank=-1, *args, **kwargs):

    if "args" in kwargs:
        kwargs = kwargs["args"]


    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    if kwargs.pre_train_mlm == True:
        output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))+"_mlm"
    else:
        output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))

    #############(for training split mlm)
    if "_s1" in kwargs.config:
        output_path += "_s1"
    elif "_s2" in kwargs.config:
        output_path += "_s2"
    else:
        pass
    #############


    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    #print("========")
    #print(parameters["valid_dataset"])
    #print(len(parameters["valid_dataset"]))
    #print("========")
    #exit()

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)


    if "projector" in kwargs.config:
        postfix = "_projector"
    elif "cross" in kwargs.config:
        postfix = "_cross"
    elif kwargs.pre_train_mlm:
        postfix = "_mlm"
    else:
        postfix = ""

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")+postfix), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")+postfix),
                exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")+postfix),
                           config.get("output", "model_name")+postfix)

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):

        #'config/restaurantPromptRoberta.config'
        config_name = kwargs.config.replace(".config","").split("/")[-1]
        prompt_dir = "model/"+config_name+"/"+str(epoch_num)+"_task_prompt.pkl"
        prompt_emb = torch.load(prompt_dir, map_location=lambda storage, loc: storage)
        prompt_emb = prompt_emb["model"]
        print("=========================")
        print("Using",prompt_dir)
        print("=========================")
        if prompt_emb != None:
            prompt_emb = torch.nn.Parameter(prompt_emb).to("cuda")

            if "Roberta" in config_name:
                model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
            elif "Bert" in config_name:
                model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
            else:
                print("Wrong!!!")
                exit()
        else:
            print("=========================")
            print("Using original prompt emb")
            print("=========================")
            pass



        start_time = timer()
        current_epoch = epoch_num
        #model.train()
        model.eval()
        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        ##############################
        #####MLM re-mask tokens#######
        ##Other task remain the same##
        ##############################
        if kwargs.pre_train_mlm == True:
            if do_test:
                test_dataset = init_test_dataset(config, *args, args=kwargs)
            else:
                parameters["train_dataset"], parameters["valid_dataset"] = init_dataset(config, *args, args=kwargs)
                dataset = parameters["train_dataset"]
        else:
            pass
        ##############################


        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            model.zero_grad()


            results = model(data, config, gpu_list, acc_result, "train", args=kwargs)
            #results = model(data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]

            total_loss += float(loss)
            #total_loss = total_loss

            #loss.backward()
            #optimizer.step()

            if step % output_time == 0 and local_rank <= 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1
            #writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)
        try:
            model.module.lower_temp(0.8)
        except:
            pass

        if local_rank <= 0:
            output_info = output_function(acc_result, config)
            delta_t = timer() - start_time
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        if local_rank <= 0:
            #checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step)
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1), current_epoch)
            ###
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch_acc", float(acc_result['right']/acc_result['total']), current_epoch)
            ###


        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function, args=kwargs)
                model.eval()
                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test")
        if local_rank >= 0:
            torch.distributed.barrier()
