#! /usr/bin/env python
"""This script handles reading command line arguments and starting the
training process. 

"""
import os
import logger
import logging
import scipy.misc
import numpy as np
import random
import sys
import time

def main(): 
    """
    Execute a complete training run.
    """
    if '-v' in sys.argv:
        VALIDATION_MODE=True
    else:
        VALIDATION_MODE=False

    screen_off= not VALIDATION_MODE

    BATCH_SIZE =256*8
    PHI_LENGTH = 1  
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    RAW_SIZE=90
    action_options = [0,  100,-100, ]
    flip_action=lambda a:-a%len(action_options)
    num_actions = len(action_options) 
    FRAME_SKIP = 4
    MAX_DATA_SET = 200000
    EPISILON_STEP=10.0/ MAX_DATA_SET
    LEARNING_RATE = 0.0004

    DISCOUNT = .97
    num_steps_before_death=50
    MOMENTUM = 0


    nn_file ='network_model'

    logging.info('build network')
    from qlearner import DeepQLearner
    network = DeepQLearner(
        RESIZED_WIDTH, RESIZED_HEIGHT,        num_actions,
        PHI_LENGTH,        DISCOUNT,        LEARNING_RATE,        MOMENTUM,
        BATCH_SIZE,)

    if len(sys.argv)>1:
        logging.info('load nn from:%s', sys.argv[1])
        network.load(sys.argv[1])
    states = np.zeros(
        (BATCH_SIZE, PHI_LENGTH, RESIZED_HEIGHT, RESIZED_WIDTH), dtype='float32')
    phi = np.zeros(
        (1, PHI_LENGTH, RESIZED_HEIGHT, RESIZED_WIDTH), dtype='float32')

    actions = np.zeros((BATCH_SIZE, 1), dtype='int32')
    rewards = np.zeros((BATCH_SIZE, 1), dtype='float32')
    next_states = np.zeros(
        (BATCH_SIZE, PHI_LENGTH, RESIZED_HEIGHT, RESIZED_WIDTH), dtype='float32')
    terminals = np.zeros((BATCH_SIZE, 1), dtype='int32')
    import chessbox as scene
    logging.info('build scene')
    scene = scene.Demo(RAW_SIZE,screen_off=screen_off)
    action = 0

    logging.info('build dataset')
    dataset = Dataset(1, MAX_DATA_SET)  
    count = 0
    t = 0
    survive = 0
    max_survive=1000
    log_rate = 30
    epsilon=0.0

    e=Experiment(size_resized=RESIZED_WIDTH,
            size_raw=RAW_SIZE,
            action_options=action_options,
            phi_length=1,
            max_dataset=int(MAX_DATA_SET/3),
            num_steps_before_death=num_steps_before_death)
    e.network=network
    e.action_options=action_options
    e.scene=scene
    e.validation_mode=VALIDATION_MODE
    self=e
    if VALIDATION_MODE:
        survive=0
        while True:
            survive+=1
            scene.renderFrame()
            img=scene.getDepthMapT()
            # get q val and action
            q_val=e.qval(img)
            action = np.argmax(q_val)
            a = self.action_options[action]
            if not 0== self.scene.step(a):
                scene.resetGame()
                logging.info('survive %s steps',survive)
                survive=0
        return 


    while True:

        count += 1
        epsilon+=EPISILON_STEP
        epsilon%=2.0

        #save network
        if count % 1000 == 10:
            network.save(nn_file)

        if True:
            scene.renderFrame()

            img=scene.getDepthMapT()

            # get action
            _r=random.random()
            if  _r>epsilon  or _r > 2-epsilon :
                action = np.random.randint(0, num_actions)
            else:
                q_val=self.qval(img)
                action = np.argmax(q_val)

            if dataset.length() < MAX_DATA_SET:
                logging.debug('dataset size:%s', dataset.length())
            a = action_options[action]
            for _ in range(FRAME_SKIP):
                reward = scene.step(a)
                if not reward == 0:
                    break

            if not 0 == reward:
                scene.resetGame()
            # add_sample to dataset
            if count > 5:
                dataset.add_sample(img, reward, action)


        #skip training
        if count % (2 * 4 * 11) == 8:  
            # get training data
            if dataset.length() > BATCH_SIZE + 5 + PHI_LENGTH:
                batches = [dataset.getRandom() for b in range(BATCH_SIZE)]
                for b in range(BATCH_SIZE):
                    s, r, a = batches[b][-2]
                    s_n, _, _ = batches[b][-1]
                    if 0 == r:
                        rewards[b][0] =0.0
                        terminals[b][0] = False
                    else:
                        rewards[b][0] = -1.0
                        terminals[b][0] = True
                    #crop 
                    before_resize=random.randint(RESIZED_WIDTH,RAW_SIZE)
                    before_resize=RESIZED_WIDTH
                    x=random.randint(0,RAW_SIZE-before_resize)
                    y=random.randint(0,RAW_SIZE-before_resize)
                    s=s[x:x+before_resize,y:y+before_resize]
                    s_n=s_n[x:x+before_resize,y:y+before_resize]
                    #resize
                    if not before_resize==RESIZED_WIDTH:
                        s=scipy.misc.imresize(s,(RESIZED_WIDTH,RESIZED_WIDTH))
                        s_n=scipy.misc.imresize(s_n,(RESIZED_WIDTH,RESIZED_WIDTH))

                    #flip
                    if random.random()>0.5:
                        s=np.fliplr(s)
                        s_n=np.fliplr(s_n)
                        a=flip_action(a)
                    actions[b][0] = a
                    states[b,0] =s
                    next_states[b,0] =s_n
                loss, q_vals = network.train(
                    states, actions, rewards, next_states, terminals)
                batchstd = network._batchstd()
                logging.info('loss:%s', loss)
                logging.info('q val mean:%s,std:%s',
                             np.mean(q_vals,axis=0), np.std(q_vals,axis=0))
                logging.info('batch std:%s', batchstd)


class Experiment:
    scene=None
    network=None
    survive_limit=100

    def __init__(self,size_raw,size_resized,action_options, phi_length, max_dataset,num_steps_before_death):
        self.num_steps_before_death=num_steps_before_death
        self.action_options=action_options
        self.size_raw =       size_raw
        self.size_resized=        size_resized
        self.phi=np.zeros((16,1,size_resized,size_resized),dtype='float32')
        self.flip_qvals=lambda qvs:np.concatenate([[qvs[0]],qvs[len(action_options)-1:(len(action_options)-1)/2:-1],qvs[(len(action_options)-1)/2:0:-1]])

        self.dataset_failed=Dataset(phi_length,max_dataset)

    def qval(self,img,tries=16):
        phi=self.phi[:tries]
        for i in range(phi.shape[0]):
            x=random.randint(0,self.size_raw-self.size_resized)
            y=random.randint(0,self.size_raw-self.size_resized)
            img_t=img[x:x+self.size_resized,y:y+self.size_resized]
            if i<phi.shape[0]/2:
                img_t=np.fliplr(img_t)
            phi[i,0]=img_t
        q_vals = self.network.q_vals(phi)
        q_val=np.zeros(q_vals.shape[1])
        for i in range(phi.shape[0]):
            if i<phi.shape[0]/2:
                q_val+=self.flip_qvals(q_vals[i])
            else:
                q_val+=q_vals[i]
        q_val/=q_vals.shape[0]
        return q_val


class Dataset:

    def __init__(self,  phi_length, max_dataset):
        self._dataset = []
        self.phi_length = phi_length
        self.max_dataset = max_dataset

    def add_sample(self, img, reward, action):
        if len(self._dataset) > self.max_dataset:
            p, r, a = self._dataset.pop()
            p[:] = img
            img = p
        self._dataset.append((img, reward, action))

    def length(self):
        return len(self._dataset)


    def getRandom(self):
        while True:
            i = random.randint(0, len(self._dataset) - self.phi_length - 1)
            l = self._dataset[i:i + self.phi_length + 1]
            for p, r, a in l[:-2]:
                if not r == 0:
                    break
            else:
                return l


if __name__ == '__main__':
    main()
