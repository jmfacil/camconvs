import tensorflow as tf
from .tfutils import *
from CAM.datareader.utils import ProgressBar

def forward_backward_step(model, inp, optimizer,loss_function, backward_step = True, 
                        fine_tunning = True, training = True,
                        add_summary=True,clip_gradients_val=100,
                        global_step=None,max_iter=1000):
    optimize_op = None
    summ = None
    trainable_variables = model.get_trainable_variables(training,fine_tunning)
    loss_function1 = lambda *a,**kw: loss_function(*a,max_iter=max_iter,global_step=global_step,**kw)
    preds,losses = model(inp, training=training,fine_tune=fine_tunning,loss_func=loss_function1)  # get output of the model.
    total_loss = losses[0]
    losses_dict = losses[1]
    _summs=[]
    if add_summary:
        if optimizer is not None:
            _summs.append(tf.summary.scalar('lr',optimizer._lr))
        for k,v in losses_dict.items():
            _summs.append(tf.summary.scalar(k,v))
        max_outputs=3
        gt_keys={'D':'depth','N':'normal'}
        print(preds)
        for i,p in enumerate(preds):
            if p is not None:
                for k,v in p.items():
                    if k == "N":
                        v=tf.clip_by_value(v,-1.,1.)
                    _summs.append(tf.summary.image("resolution_"+str(i)+"/"+k+"/pred",
                        v,max_outputs=max_outputs))
                    if k == "D" or k == "N":
                        gt=inp['gt'][gt_keys[k]+str(i)]
                        _summs.append(tf.summary.image("resolution_"+str(i)+"/"+k+"/gt",
                        gt,max_outputs=max_outputs))
    summ  = tf.summary.merge_all()
    if backward_step:
        grads_and_vars = optimizer.compute_gradients(loss=total_loss, var_list=trainable_variables, 
                                                    colocate_gradients_with_ops=False)
        clipped_grads_and_vars = []
        with tf.name_scope('clip_gradients'):
            for g, v in grads_and_vars:
                if not g is None:
                    clipped_g = tf.clip_by_value(g, clip_value_min=-clip_gradients_val, clip_value_max=clip_gradients_val)
                    clipped_grads_and_vars.append((clipped_g,v))
                else:
                    clipped_grads_and_vars.append((g,v))
        extra_ops =tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            optimize_op = optimizer.apply_gradients(grads_and_vars=clipped_grads_and_vars,
                                                global_step=global_step)# Apply gradientes

        

    l = [optimize_op,total_loss,losses_dict, summ]
    return tuple([e for e in l if e is not None])

_e=1000*43

def mainloop(session,it,max_iter,global_stepf,
            train_ops, test_ops,
            show_progress=True, show_loss=True,add_summary=True, test = True,
            check_every = 200, save_every = 1000,
            summs = None):
    print("...")
    real_check_every = 1
    progress_bar = ProgressBar(msg='Loading network... ',
                waiting_char='█', length = 11,end_msg = 'COMPLETED',total=max_iter)
    progress_bar.start()
    def do_checks():
        _,total_loss,losses_dict, summ,gs = session.run(list(train_ops)+[global_stepf])
        ttotal_loss,tlosses_dict, tsumm = session.run(list(test_ops))
        text = '[It:{: 6.0f}]'.format(gs)+' TRAIN: {: 6.3f}'.format(total_loss)+' TEST: {: 6.3f}'.format(ttotal_loss)
        progress_bar.msg = text
        progress_bar.update(it)
        if summs is not None:
            summs[0].add_summary(summ,it)
            summs[1].add_summary(tsumm,it)
        #print(text)
    def do_save():
        print('')
        print('SAVING...')
    while it < max_iter:
        if it%save_every==0:
            do_save()
        if it%real_check_every==0:

            do_checks()
            real_check_every = min(real_check_every*2,check_every)
        else:
            _,total_loss=session.run([train_ops[0],train_ops[1]])
            #text = 'It:{: 6.0f}'.format(float(it))+' TRAIN: {: 6.3f}'.format(total_loss)+' TEST: {: 6.3f}'.format(ttotal_loss)
            #progress_bar.msg = text
            progress_bar.update(it)
        
        
        it = it + 1
    progress_bar.finish_wait()

# Trains the model for certains epochs on a dataset
def train(input_train,input_test, optimizer_func, loss_function, model, 
        iters=[1*_e,1*_e,2*_e,5*_e], 
        tr=['LR','MR','HR',True],
        ft = [False,False,False,True],
        init_lr=[5e-3,2e-3,1e-3,1e-3], min_lr = 1e-6,
        batch_size=16,
        evaluation=True, name_best_model = 'weights/best',
        temp_backup = 'temp_backup/',
        summ_folder='temp_summary/',):
    
    

    ev = 0
    restore_previous = None
    import datetime
    import os
    date = datetime.datetime.now().strftime("%Y-%m-%d(%H:%M)") 
    traindir = os.path.join(summ_folder,date+'_train')    
    testdir = os.path.join(summ_folder,date+'_test')   
    train_summary_writer = tf.summary.FileWriter(traindir)
    test_summary_writer = tf.summary.FileWriter(testdir)  
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    it_zeros = {'image':tf.zeros((8,192,256,3)),
             'intrinsics':tf.ones((8,1,4))}
    _=model(it_zeros,training=False,loss_func=None)        
    for training,fine_tunning,base_lr,max_iter in zip(tr,ft,init_lr,iters):
        ev = ev + 1
        with tf.name_scope('evolution_{}'.format(ev)):
            #Setting Learning Rate Policy
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            global_stepf = tf.cast(global_step_tensor,tf.float32)
            lr = ease_in_quad(global_stepf-max_iter/3,base_lr, 
                        min_lr-base_lr,float(2*max_iter/3) )

            # Creating operations
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_ops = forward_backward_step(model, input_train, optimizer,loss_function, backward_step = True, 
                            fine_tunning = fine_tunning, training = training,global_step=global_step_tensor,max_iter=max_iter)
            test_ops = forward_backward_step(model, input_test, optimizer,loss_function, backward_step = False, 
                            fine_tunning = False, training = training,add_summary=False)
            
            if restore_previous is None:
                init_op = tf.global_variables_initializer()
                sess.run(init_op)

            else:
                # Load state
                pass
            it = tf.train.global_step(sess, global_step_tensor)
            mainloop(sess,it,max_iter,global_stepf,train_ops,test_ops,
                summs=(train_summary_writer,test_summary_writer))
            




# Trains the model for certains epochs on a dataset
def train1(input, optimizer, loss_function, model, size_input,  epochs=5, batch_size=2, lr=None, init_lr=2e-4,
          evaluation=True, name_best_model = 'weights/best', preprocess_mode=None, labels_resize_factor=1):
    # Parameters for training
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0
    log_freq = min(50, int(steps_per_epoch/5))
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train') # tensorboard
    test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test') # tensorboard
    print('Please enter in terminal: tensorboard --logdir \\tmp\\summaries')

    for epoch in range(epochs):  # for each epoch
        lr_decay(lr, init_lr, 1e-9, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch+1) + '. Learning rate: ' + str(lr.numpy()))
        for step in range(steps_per_epoch):  # for every batch
            # get batch
            x, y, mask = loader.get_batch(size=batch_size, train=True)
            x = preprocess(x, mode=preprocess_mode)

            with train_summary_writer.as_default():
                loss = train_step(model, x, y, mask, loss_function, optimizer, labels_resize_factor, size_input)
                # tensorboard
                avg_loss.update_state(loss)
                if tf.equal(optimizer.iterations % log_freq, 0):
                    tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                    avg_loss.reset_states()


        if evaluation:
            # get metrics
            with train_summary_writer.as_default():
                train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode,
                                                    labels_resize_factor=labels_resize_factor, optimizer=optimizer)

            with test_summary_writer.as_default():
                test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                                  scales=[1], preprocess_mode=preprocess_mode, labels_resize_factor=labels_resize_factor, optimizer=optimizer)

            print('Train accuracy: ' + str(train_acc.numpy()))
            print('Train miou: ' + str(train_miou.numpy()))
            print('Test accuracy: ' + str(test_acc.numpy()))
            print('Test miou: ' + str(test_miou.numpy()))
            print('')

            # save model if bet
            if test_miou > best_miou:
                best_miou = test_miou
                model.save_weights(name_best_model)
        else:
            model.save_weights(name_best_model)

        loader.suffle_segmentation()  # sheffle trainign set
