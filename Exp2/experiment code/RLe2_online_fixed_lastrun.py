#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.4),
    on Fri Mar  3 15:11:36 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from choose_code
trialsN = 0
import numpy as np
import random
#vnml=np.random.normal(loc=0,scale=2.8,size=150)
#vzhengshu=np.round(vnml)#生成噪声v的高斯分布数列并四舍五入取整（same with原论文
#vshuzi=vzhengshu.tolist()
#v1=random.sample(vshuzi,1)
#v=v1[0]
#tempArray = ['bandit0']
#random.shuffle(tempArray)
#expect_which = locals()[tempArray[0]]

class delta_rule():
    def __init__(self,beta=0.04,lr=0.5):
        self.Q_value_table = np.ones((1,4)).squeeze()*50
        self.ACTION = np.array([0,1,2,3])
        self.beta = beta
        self.lr = lr
    def P_action_softmax(self,action):
        return np.exp(self.Q_value_table[action]*self.beta)/(np.exp(self.Q_value_table[0]*self.beta)+np.exp(self.Q_value_table[1]*self.beta)+np.exp(self.Q_value_table[2]*self.beta)+np.exp(self.Q_value_table[3]*self.beta))
    
    def sample_softmax(self):
        Prob = np.array([self.P_action_softmax(0),self.P_action_softmax(1),self.P_action_softmax(2),self.P_action_softmax(3)])
        return np.random.choice(a=self.ACTION,size=1,replace=True,p=Prob)
    def learn(self,action,reward):
        self.Q_value_table[action] = self.Q_value_table[action]+self.lr*(reward-self.Q_value_table[action])

lists = [random.uniform(25,45),random.uniform(35,55) ,random.uniform(45,65) ,random.uniform(55,75)]

random.shuffle(lists)

uyestart = lists[0]
ubluestart = lists[1]
ugreenstart = lists[2]
uredstart = lists[3]

#均值起始值
B_start = [random.uniform(25,45) ,random.uniform(35,55) ,random.uniform(45,65) ,random.uniform(55,75)]

decaypara=0.9836
decaycenter=50

B_agent = delta_rule(beta=0.06,lr=0.5)



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.4'
expName = 'RL'  # from the Builder filename that created this script
expInfo = {
    'participant': '1',
    'Gender': 'f',
    'Age': '1',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s' % (expInfo['participant'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/wututu/Library/CloudStorage/OneDrive-UniversityofMacau/RL_E2/RLe2_online_fixed_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}
ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='ptb')

# --- Initialize components for Routine "instru" ---
instru_img1 = visual.ImageStim(
    win=win,
    name='instru_img1', units='height', 
    image='pic/instru1.png', mask=None, anchor='center',
    ori=0, pos=(0, 0), size=(1.56,0.8),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
instru_resp = keyboard.Keyboard()

# --- Initialize components for Routine "instr2" ---
instru_img2 = visual.ImageStim(
    win=win,
    name='instru_img2', 
    image='pic/instru2.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.56,0.9),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
instru_resp2 = keyboard.Keyboard()

# --- Initialize components for Routine "instru3" ---
image_6 = visual.ImageStim(
    win=win,
    name='image_6', 
    image='pic/instru3.png', mask=None, anchor='center',
    ori=0.0, pos=(0,0), size=(1.53, 1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
key_resp_2 = keyboard.Keyboard()

# --- Initialize components for Routine "fixbefore" ---
fixbefore_2 = visual.TextStim(win=win, name='fixbefore_2',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "rest" ---
rest_image = visual.ImageStim(
    win=win,
    name='rest_image', 
    image='pic/rest.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(0.857, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
rest_response = keyboard.Keyboard()

# --- Initialize components for Routine "choose" ---
red = visual.ImageStim(
    win=win,
    name='red', 
    image='pic/red.png', mask=None, anchor='center',
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
green = visual.ImageStim(
    win=win,
    name='green', 
    image='pic/green.png', mask=None, anchor='center',
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
yellow = visual.ImageStim(
    win=win,
    name='yellow', 
    image='pic/yellow.png', mask=None, anchor='center',
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
blue = visual.ImageStim(
    win=win,
    name='blue', 
    image='pic/blue.png', mask=None, anchor='center',
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
choose_bandit = keyboard.Keyboard()
Fixation = visual.TextStim(win=win, name='Fixation',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);

# --- Initialize components for Routine "choose_result" ---
black = visual.ImageStim(
    win=win,
    name='black', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=[0,0], size=[0.33],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
red_2 = visual.ImageStim(
    win=win,
    name='red_2', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
green_2 = visual.ImageStim(
    win=win,
    name='green_2', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
yellow_2 = visual.ImageStim(
    win=win,
    name='yellow_2', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
blue_2 = visual.ImageStim(
    win=win,
    name='blue_2', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
fixation_2 = visual.TextStim(win=win, name='fixation_2',
    text='',
    font='Arial',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-6.0);
docs = visual.TextStim(win=win, name='docs',
    text='',
    font='Arial',
    pos=[0,0], height=0.05, wrapWidth=None, ori=0, 
    color='black', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-7.0);
image_2 = visual.ImageStim(
    win=win,
    name='image_2', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.3), size=(0.51, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)

# --- Initialize components for Routine "comparison_result" ---
black_com = visual.ImageStim(
    win=win,
    name='black_com', 
    image='pic/black.png', mask=None, anchor='center',
    ori=0, pos=[0,0], size=[0.33],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
reward_sub = visual.ImageStim(
    win=win,
    name='reward_sub', 
    image='pic/rewardsub.png', mask=None, anchor='center',
    ori=0.0, pos=(-0.215, 0.3), size=(0.365,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
palyerb = visual.ImageStim(
    win=win,
    name='palyerb', 
    image='pic/rewardpB.png', mask=None, anchor='center',
    ori=0.0, pos=(0.215, 0.3), size=(0.427,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
reward__sub = visual.TextStim(win=win, name='reward__sub',
    text='',
    font='Open Sans',
    pos=(-0.15, 0.305), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);
compare = visual.TextStim(win=win, name='compare',
    text='',
    font='Open Sans',
    pos=(0.321, 0.305), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-5.0);
red_compare = visual.ImageStim(
    win=win,
    name='red_compare', 
    image='pic/red.png', mask=None, anchor='center',
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-6.0)
green_com = visual.ImageStim(
    win=win,
    name='green_com', 
    image='pic/green.png', mask=None, anchor='center',
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-7.0)
yellow_com = visual.ImageStim(
    win=win,
    name='yellow_com', 
    image='pic/yellow.png', mask=None, anchor='center',
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-8.0)
blue_com = visual.ImageStim(
    win=win,
    name='blue_com', 
    image='pic/blue.png', mask=None, anchor='center',
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-9.0)

# --- Initialize components for Routine "happy_rating" ---
image_3 = visual.ImageStim(
    win=win,
    name='image_3', 
    image='pic/happychoose.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.2), size=(0.57, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
slider = visual.Slider(win=win, name='slider',
    startValue=None, size=(1.0, 0.1), pos=(0, 0), units=None,
    labels=None, ticks=(1,2,3,4,5,6,7), granularity=1.0,
    style='rating', styleTweaks=(), opacity=None,
    labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, ori=0.0, depth=-2, readOnly=False)
text = visual.TextStim(win=win, name='text',
    text=None,
    font='Open Sans',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
image_4 = visual.ImageStim(
    win=win,
    name='image_4', 
    image='pic/happy.png', mask=None, anchor='center',
    ori=0.0, pos=(0.7, 0), size=(0.183, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)
image_5 = visual.ImageStim(
    win=win,
    name='image_5', 
    image='pic/sad.png', mask=None, anchor='center',
    ori=0.0, pos=(-0.7, 0), size=(0.217, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-5.0)

# --- Initialize components for Routine "expect" ---
expect_self_black = visual.ImageStim(
    win=win,
    name='expect_self_black', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=[0,0], size=[0.33],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
expect_self = visual.ImageStim(
    win=win,
    name='expect_self', 
    image='pic/expect_self.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.3), size=(1, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
expectself = visual.TextBox2(
     win, text=None, font='Open Sans',
     pos=(0.10, 0.25),     letterHeight=0.05,
     size=(None, None), borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0, alignment='center',
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False, languageStyle='Arabic',
     editable=True,
     name='expectself',
     autoLog=True,
)
expect_self_red = visual.ImageStim(
    win=win,
    name='expect_self_red', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
expect_self_green = visual.ImageStim(
    win=win,
    name='expect_self_green', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
expect_self_yellow = visual.ImageStim(
    win=win,
    name='expect_self_yellow', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-6.0)
expect_self_blue = visual.ImageStim(
    win=win,
    name='expect_self_blue', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-7.0)
tijiao = visual.ImageStim(
    win=win,
    name='tijiao', 
    image='pic/tijiao.png', mask=None, anchor='center',
    ori=0.0, pos=(0, -0.3), size=(0.38, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)
tijiao_click = event.Mouse(win=win)
x, y = [None, None]
tijiao_click.mouseClock = core.Clock()

# --- Initialize components for Routine "expect_random" ---
jiantou_random = visual.ImageStim(
    win=win,
    name='jiantou_random', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=[0,0], size=(0.077,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
expect_red = visual.ImageStim(
    win=win,
    name='expect_red', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
expect_green = visual.ImageStim(
    win=win,
    name='expect_green', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
expect_yellow = visual.ImageStim(
    win=win,
    name='expect_yellow', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
expect_blue = visual.ImageStim(
    win=win,
    name='expect_blue', 
    image='sin', mask=None, anchor='center',
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
expect_random_2 = visual.ImageStim(
    win=win,
    name='expect_random_2', 
    image='pic/random_expect.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.35), size=(0.72, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-6.0)
expect_random_3 = visual.TextBox2(
     win, text=None, font='Open Sans',
     pos=(0.1, 0.3),     letterHeight=0.05,
     size=(None, None), borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0, alignment='center',
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False, languageStyle='Arabic',
     editable=True,
     name='expect_random_3',
     autoLog=True,
)
tijiao_random = event.Mouse(win=win)
x, y = [None, None]
tijiao_random.mouseClock = core.Clock()
tijiao_random_2 = visual.ImageStim(
    win=win,
    name='tijiao_random_2', 
    image='pic/tijiao.png', mask=None, anchor='center',
    ori=0.0, pos=(0, -0.3), size=(0.38, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-9.0)

# --- Initialize components for Routine "confidence" ---
slider_2 = visual.Slider(win=win, name='slider_2',
    startValue=None, size=(1.0, 0.1), pos=(0, 0), units=None,
    labels=(1, 2, 3, 4, 5,6,7,8,9,10,11), ticks=(1, 2, 3, 4, 5,6,7,8,9,10,11), granularity=0.0,
    style='rating', styleTweaks=(), opacity=None,
    labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, ori=0.0, depth=0, readOnly=False)
image_7 = visual.ImageStim(
    win=win,
    name='image_7', 
    image='pic/confi.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0.2), size=(0.67, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
confi = visual.ImageStim(
    win=win,
    name='confi', 
    image='pic/confiden.png', mask=None, anchor='center',
    ori=0.0, pos=(0.7, 0), size=(0.229, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
notconfi = visual.ImageStim(
    win=win,
    name='notconfi', 
    image='pic/notconfi.png', mask=None, anchor='center',
    ori=0.0, pos=(-0.7, -0.01), size=(0.252, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)

# --- Initialize components for Routine "expectB" ---
image_8 = visual.ImageStim(
    win=win,
    name='image_8', 
    image='pic/expect_B.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.19, 0.3),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
expectB_2 = visual.TextBox2(
     win, text=None, font='Open Sans',
     pos=(0.15, -0.07),     letterHeight=0.05,
     size=(None, None), borderWidth=2.0,
     color='white', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0, alignment='center',
     anchor='center',
     fillColor=None, borderColor=None,
     flipHoriz=False, flipVert=False, languageStyle='LTR',
     editable=True,
     name='expectB_2',
     autoLog=True,
)
tijiao_B_2 = visual.ImageStim(
    win=win,
    name='tijiao_B_2', 
    image='pic/tijiao.png', mask=None, anchor='center',
    ori=0.0, pos=(0, -0.3), size=(0.38, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
tijiao_B = event.Mouse(win=win)
x, y = [None, None]
tijiao_B.mouseClock = core.Clock()

# --- Initialize components for Routine "wrongsign" ---
bigx = visual.TextStim(win=win, name='bigx',
    text='',
    font='Arial',
    pos=(0, 0), height=0.5, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# --- Initialize components for Routine "intertrial" ---
intertrialfix = visual.TextStim(win=win, name='intertrialfix',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "end" ---
image = visual.ImageStim(
    win=win,
    name='image', 
    image='pic/end.png', mask=None, anchor='center',
    ori=0, pos=(0, 0), size=(0.661,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
key_resp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "instru" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
instru_resp.keys = []
instru_resp.rt = []
_instru_resp_allKeys = []
# keep track of which components have finished
instruComponents = [instru_img1, instru_resp]
for thisComponent in instruComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "instru" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instru_img1* updates
    if instru_img1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instru_img1.frameNStart = frameN  # exact frame index
        instru_img1.tStart = t  # local t and not account for scr refresh
        instru_img1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instru_img1, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'instru_img1.started')
        instru_img1.setAutoDraw(True)
    
    # *instru_resp* updates
    waitOnFlip = False
    if instru_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instru_resp.frameNStart = frameN  # exact frame index
        instru_resp.tStart = t  # local t and not account for scr refresh
        instru_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instru_resp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'instru_resp.started')
        instru_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instru_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instru_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instru_resp.status == STARTED and not waitOnFlip:
        theseKeys = instru_resp.getKeys(keyList=['space'], waitRelease=False)
        _instru_resp_allKeys.extend(theseKeys)
        if len(_instru_resp_allKeys):
            instru_resp.keys = _instru_resp_allKeys[-1].name  # just the last key pressed
            instru_resp.rt = _instru_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "instru" ---
for thisComponent in instruComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instru_resp.keys in ['', [], None]:  # No response was made
    instru_resp.keys = None
thisExp.addData('instru_resp.keys',instru_resp.keys)
if instru_resp.keys != None:  # we had a response
    thisExp.addData('instru_resp.rt', instru_resp.rt)
thisExp.nextEntry()
# the Routine "instru" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "instr2" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
instru_resp2.keys = []
instru_resp2.rt = []
_instru_resp2_allKeys = []
# keep track of which components have finished
instr2Components = [instru_img2, instru_resp2]
for thisComponent in instr2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "instr2" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instru_img2* updates
    if instru_img2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instru_img2.frameNStart = frameN  # exact frame index
        instru_img2.tStart = t  # local t and not account for scr refresh
        instru_img2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instru_img2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'instru_img2.started')
        instru_img2.setAutoDraw(True)
    
    # *instru_resp2* updates
    waitOnFlip = False
    if instru_resp2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instru_resp2.frameNStart = frameN  # exact frame index
        instru_resp2.tStart = t  # local t and not account for scr refresh
        instru_resp2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instru_resp2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'instru_resp2.started')
        instru_resp2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instru_resp2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instru_resp2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instru_resp2.status == STARTED and not waitOnFlip:
        theseKeys = instru_resp2.getKeys(keyList=['s'], waitRelease=False)
        _instru_resp2_allKeys.extend(theseKeys)
        if len(_instru_resp2_allKeys):
            instru_resp2.keys = _instru_resp2_allKeys[-1].name  # just the last key pressed
            instru_resp2.rt = _instru_resp2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "instr2" ---
for thisComponent in instr2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instru_resp2.keys in ['', [], None]:  # No response was made
    instru_resp2.keys = None
thisExp.addData('instru_resp2.keys',instru_resp2.keys)
if instru_resp2.keys != None:  # we had a response
    thisExp.addData('instru_resp2.rt', instru_resp2.rt)
thisExp.nextEntry()
# the Routine "instr2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "instru3" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
key_resp_2.keys = []
key_resp_2.rt = []
_key_resp_2_allKeys = []
# keep track of which components have finished
instru3Components = [image_6, key_resp_2]
for thisComponent in instru3Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "instru3" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *image_6* updates
    if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        image_6.frameNStart = frameN  # exact frame index
        image_6.tStart = t  # local t and not account for scr refresh
        image_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'image_6.started')
        image_6.setAutoDraw(True)
    
    # *key_resp_2* updates
    waitOnFlip = False
    if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_2.frameNStart = frameN  # exact frame index
        key_resp_2.tStart = t  # local t and not account for scr refresh
        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'key_resp_2.started')
        key_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_2.getKeys(keyList=['y'], waitRelease=False)
        _key_resp_2_allKeys.extend(theseKeys)
        if len(_key_resp_2_allKeys):
            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instru3Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "instru3" ---
for thisComponent in instru3Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_2.keys in ['', [], None]:  # No response was made
    key_resp_2.keys = None
thisExp.addData('key_resp_2.keys',key_resp_2.keys)
if key_resp_2.keys != None:  # we had a response
    thisExp.addData('key_resp_2.rt', key_resp_2.rt)
thisExp.nextEntry()
# the Routine "instru3" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "fixbefore" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
# keep track of which components have finished
fixbeforeComponents = [fixbefore_2]
for thisComponent in fixbeforeComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "fixbefore" ---
while continueRoutine and routineTimer.getTime() < 1.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *fixbefore_2* updates
    if fixbefore_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        fixbefore_2.frameNStart = frameN  # exact frame index
        fixbefore_2.tStart = t  # local t and not account for scr refresh
        fixbefore_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(fixbefore_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'fixbefore_2.started')
        fixbefore_2.setAutoDraw(True)
    if fixbefore_2.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > fixbefore_2.tStartRefresh + 1-frameTolerance:
            # keep track of stop time/frame for later
            fixbefore_2.tStop = t  # not accounting for scr refresh
            fixbefore_2.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixbefore_2.stopped')
            fixbefore_2.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in fixbeforeComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "fixbefore" ---
for thisComponent in fixbeforeComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-1.000000)

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=150, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('sequence.xlsx'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "rest" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    rest_response.keys = []
    rest_response.rt = []
    _rest_response_allKeys = []
    # Run 'Begin Routine' code from code_rest
    if trialsN == 0 or trialsN % 75 != 0:
        continueRoutine = False
    
    # keep track of which components have finished
    restComponents = [rest_image, rest_response]
    for thisComponent in restComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "rest" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rest_image* updates
        if rest_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_image.frameNStart = frameN  # exact frame index
            rest_image.tStart = t  # local t and not account for scr refresh
            rest_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_image.started')
            rest_image.setAutoDraw(True)
        
        # *rest_response* updates
        waitOnFlip = False
        if rest_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_response.frameNStart = frameN  # exact frame index
            rest_response.tStart = t  # local t and not account for scr refresh
            rest_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_response, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_response.started')
            rest_response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(rest_response.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(rest_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if rest_response.status == STARTED and not waitOnFlip:
            theseKeys = rest_response.getKeys(keyList=['q'], waitRelease=False)
            _rest_response_allKeys.extend(theseKeys)
            if len(_rest_response_allKeys):
                rest_response.keys = _rest_response_allKeys[-1].name  # just the last key pressed
                rest_response.rt = _rest_response_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in restComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in restComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if rest_response.keys in ['', [], None]:  # No response was made
        rest_response.keys = None
    trials.addData('rest_response.keys',rest_response.keys)
    if rest_response.keys != None:  # we had a response
        trials.addData('rest_response.rt', rest_response.rt)
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "choose" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    choose_bandit.keys = []
    choose_bandit.rt = []
    _choose_bandit_allKeys = []
    # Run 'Begin Routine' code from choose_code
    import random
    #vnml=np.random.normal(loc=0,scale=2.8,size=150)
    #vzhengshu=np.round(vnml)#生成噪声v的高斯分布数列并四舍五入取整（same with原论文
    #vshuzi=vzhengshu.tolist()
    #v1=random.sample(vshuzi,1)
    #v=v1[0]
    #tempArray = ['bandit0','black_x_pos_random','black_y_pos_random']
    #random.shuffle(tempArray)
    #expect_which = locals()[tempArray[0]]
    #black_x_random= locals()[tempArray[1]]
    #black_y_random = locals()[tempArray[2]]
    
    import random
    expect_which = random.choice([0,1])
    black_x_random= random.choice([-0.6,-0.2,0.2,0.6])
    #black_y_random = random.choice([-0.225,0.225])
    # keep track of which components have finished
    chooseComponents = [red, green, yellow, blue, choose_bandit, Fixation]
    for thisComponent in chooseComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "choose" ---
    while continueRoutine and routineTimer.getTime() < 1.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *red* updates
        if red.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            red.frameNStart = frameN  # exact frame index
            red.tStart = t  # local t and not account for scr refresh
            red.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(red, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'red.started')
            red.setAutoDraw(True)
        if red.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > red.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                red.tStop = t  # not accounting for scr refresh
                red.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red.stopped')
                red.setAutoDraw(False)
        
        # *green* updates
        if green.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green.frameNStart = frameN  # exact frame index
            green.tStart = t  # local t and not account for scr refresh
            green.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'green.started')
            green.setAutoDraw(True)
        if green.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > green.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                green.tStop = t  # not accounting for scr refresh
                green.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green.stopped')
                green.setAutoDraw(False)
        
        # *yellow* updates
        if yellow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            yellow.frameNStart = frameN  # exact frame index
            yellow.tStart = t  # local t and not account for scr refresh
            yellow.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellow, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'yellow.started')
            yellow.setAutoDraw(True)
        if yellow.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > yellow.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                yellow.tStop = t  # not accounting for scr refresh
                yellow.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow.stopped')
                yellow.setAutoDraw(False)
        
        # *blue* updates
        if blue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blue.frameNStart = frameN  # exact frame index
            blue.tStart = t  # local t and not account for scr refresh
            blue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blue.started')
            blue.setAutoDraw(True)
        if blue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blue.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                blue.tStop = t  # not accounting for scr refresh
                blue.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue.stopped')
                blue.setAutoDraw(False)
        
        # *choose_bandit* updates
        waitOnFlip = False
        if choose_bandit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choose_bandit.frameNStart = frameN  # exact frame index
            choose_bandit.tStart = t  # local t and not account for scr refresh
            choose_bandit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choose_bandit, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'choose_bandit.started')
            choose_bandit.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(choose_bandit.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(choose_bandit.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if choose_bandit.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > choose_bandit.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                choose_bandit.tStop = t  # not accounting for scr refresh
                choose_bandit.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choose_bandit.stopped')
                choose_bandit.status = FINISHED
        if choose_bandit.status == STARTED and not waitOnFlip:
            theseKeys = choose_bandit.getKeys(keyList=['r', 'f', 'i', 'j'], waitRelease=False)
            _choose_bandit_allKeys.extend(theseKeys)
            if len(_choose_bandit_allKeys):
                choose_bandit.keys = _choose_bandit_allKeys[-1].name  # just the last key pressed
                choose_bandit.rt = _choose_bandit_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *Fixation* updates
        if Fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Fixation.frameNStart = frameN  # exact frame index
            Fixation.tStart = t  # local t and not account for scr refresh
            Fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Fixation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Fixation.started')
            Fixation.setAutoDraw(True)
        if Fixation.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Fixation.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                Fixation.tStop = t  # not accounting for scr refresh
                Fixation.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fixation.stopped')
                Fixation.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in chooseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "choose" ---
    for thisComponent in chooseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if choose_bandit.keys in ['', [], None]:  # No response was made
        choose_bandit.keys = None
    trials.addData('choose_bandit.keys',choose_bandit.keys)
    if choose_bandit.keys != None:  # we had a response
        trials.addData('choose_bandit.rt', choose_bandit.rt)
    # Run 'End Routine' code from choose_code
    trialsN = trialsN + 1
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.500000)
    
    # --- Prepare to start Routine "choose_result" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from result_code
    uyellow = uyestart*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    ured = uredstart*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    ublue = ubluestart*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    ugreen = ugreenstart*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    B_0 = B_start[0]*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    B_1 = B_start[1]*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    B_2 = B_start[2]*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    B_3 = B_start[3]*decaypara+(1-decaypara)*decaycenter + np.random.normal(0,2.8,1)
    import random
    if trialsN == 1:
        rewardyellow=np.random.normal(uyestart,4,1)
        rewardred=np.random.normal(uredstart,4,1)
        rewardblue=np.random.normal(ubluestart,4,1)
        rewardgreen=np.random.normal(ugreenstart,4,1)
        #trial 1 时是初始值，所以不需要加噪声。这里的1是从这个分布里取出“1”个数字的意思
    else:
        rewardyellow=np.random.normal(uyellow,4,1)
        rewardred=np.random.normal(ured,4,1)
        rewardblue=np.random.normal(ublue,4,1)
        rewardgreen=np.random.normal(ugreen,4,1)
    #定义每个bandit的reward
    z = np.random.uniform(0.8,2)#生成随机小数作为result的duration
    f = z+0.3#图片呈现时间
    
    if choose_bandit.keys == 'r':#如果被试选择某一个bandit，就显示对应的值
        black_x_pos = -0.6
        black_y_pos = 0.
        docs_x_pos = -0.6
        docs_y_pos = 0
        points = int(np.round(rewardyellow))
        ss = 300
    elif choose_bandit.keys == 'f':
        black_x_pos = -0.2
        black_y_pos = 0
        docs_x_pos = -0.2
        docs_y_pos = 0
        ss = 300
        points = int(np.round(rewardred))
    elif choose_bandit.keys == 'i':
        black_x_pos = 0.2
        black_y_pos = 0
        docs_x_pos = 0.2
        docs_y_pos = 0
        ss = 300
        points = int(np.round(rewardblue))
    elif choose_bandit.keys == 'j':
        black_x_pos = 0.6
        black_y_pos = 0
        docs_x_pos = 0.6
        docs_y_pos = 0
        ss = 300
        points = int(np.round(rewardgreen))
    else:
        black_x_pos = 9
        black_y_pos = 9
        ss = 0
        continueRoutine = False
    
    fix_time = 1.5
    rewardyellowlist=[]
    rewardredlist=[]
    rewardbluelist=[]
    rewardgreenlist=[]
    rewardyellowlist.append(int(np.round(rewardyellow)))
    rewardredlist.append(int(np.round(rewardred)))
    rewardbluelist.append(int(np.round(rewardblue)))
    rewardgreenlist.append(int(np.round(rewardgreen)))
    
    black.setPos((black_x_pos, black_y_pos))
    black.setImage('pic/black.png')
    red_2.setImage('pic/red.png')
    green_2.setImage('pic/green.png')
    yellow_2.setImage('pic/yellow.png')
    blue_2.setImage('pic/blue.png')
    fixation_2.setText('+')
    docs.setPos((docs_x_pos,docs_y_pos))
    docs.setText('● ● ●')
    image_2.setImage('pic/waitplayerB.png')
    # keep track of which components have finished
    choose_resultComponents = [black, red_2, green_2, yellow_2, blue_2, fixation_2, docs, image_2]
    for thisComponent in choose_resultComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "choose_result" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *black* updates
        if black.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            black.frameNStart = frameN  # exact frame index
            black.tStart = t  # local t and not account for scr refresh
            black.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(black, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'black.started')
            black.setAutoDraw(True)
        if black.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > black.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                black.tStop = t  # not accounting for scr refresh
                black.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black.stopped')
                black.setAutoDraw(False)
        
        # *red_2* updates
        if red_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            red_2.frameNStart = frameN  # exact frame index
            red_2.tStart = t  # local t and not account for scr refresh
            red_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(red_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'red_2.started')
            red_2.setAutoDraw(True)
        if red_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > red_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                red_2.tStop = t  # not accounting for scr refresh
                red_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_2.stopped')
                red_2.setAutoDraw(False)
        
        # *green_2* updates
        if green_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green_2.frameNStart = frameN  # exact frame index
            green_2.tStart = t  # local t and not account for scr refresh
            green_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'green_2.started')
            green_2.setAutoDraw(True)
        if green_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > green_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                green_2.tStop = t  # not accounting for scr refresh
                green_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_2.stopped')
                green_2.setAutoDraw(False)
        
        # *yellow_2* updates
        if yellow_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            yellow_2.frameNStart = frameN  # exact frame index
            yellow_2.tStart = t  # local t and not account for scr refresh
            yellow_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellow_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'yellow_2.started')
            yellow_2.setAutoDraw(True)
        if yellow_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > yellow_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                yellow_2.tStop = t  # not accounting for scr refresh
                yellow_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow_2.stopped')
                yellow_2.setAutoDraw(False)
        
        # *blue_2* updates
        if blue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blue_2.frameNStart = frameN  # exact frame index
            blue_2.tStart = t  # local t and not account for scr refresh
            blue_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blue_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blue_2.started')
            blue_2.setAutoDraw(True)
        if blue_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blue_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                blue_2.tStop = t  # not accounting for scr refresh
                blue_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_2.stopped')
                blue_2.setAutoDraw(False)
        
        # *fixation_2* updates
        if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_2.frameNStart = frameN  # exact frame index
            fixation_2.tStart = t  # local t and not account for scr refresh
            fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_2.started')
            fixation_2.setAutoDraw(True)
        if fixation_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                fixation_2.tStop = t  # not accounting for scr refresh
                fixation_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2.stopped')
                fixation_2.setAutoDraw(False)
        
        # *docs* updates
        if docs.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
            # keep track of start time/frame for later
            docs.frameNStart = frameN  # exact frame index
            docs.tStart = t  # local t and not account for scr refresh
            docs.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(docs, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'docs.started')
            docs.setAutoDraw(True)
        if docs.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > docs.tStartRefresh + z-frameTolerance:
                # keep track of stop time/frame for later
                docs.tStop = t  # not accounting for scr refresh
                docs.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'docs.stopped')
                docs.setAutoDraw(False)
        
        # *image_2* updates
        if image_2.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            image_2.setAutoDraw(True)
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + z-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.stopped')
                image_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in choose_resultComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "choose_result" ---
    for thisComponent in choose_resultComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "choose_result" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "comparison_result" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from comparisonresult_code
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        continueRoutine = True
    else:
        continueRoutine = False
    import numpy as np
    B_action = B_agent.sample_softmax()
    if B_action == 0:
        B_reward = int(np.random.normal(B_0,4,1))
    elif B_action == 1:
        B_reward = int(np.random.normal(B_1,4,1))
    elif B_action == 2:
        B_reward = int(np.random.normal(B_2,4,1))
    else :
        B_reward = int(np.random.normal(B_3,4,1))
    
    B_agent.learn(B_action,B_reward)
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        a = points
        j = B_reward
    else:
        a = 0
        j = '**'
    jlist=[]
    alist=[]
    jlist.append(j)
    alist.append(a)
    black_com.setPos((black_x_pos, black_y_pos))
    reward__sub.setText(a)
    compare.setText(j)
    # keep track of which components have finished
    comparison_resultComponents = [black_com, reward_sub, palyerb, reward__sub, compare, red_compare, green_com, yellow_com, blue_com]
    for thisComponent in comparison_resultComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "comparison_result" ---
    while continueRoutine and routineTimer.getTime() < 2.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *black_com* updates
        if black_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            black_com.frameNStart = frameN  # exact frame index
            black_com.tStart = t  # local t and not account for scr refresh
            black_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(black_com, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'black_com.started')
            black_com.setAutoDraw(True)
        if black_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > black_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                black_com.tStop = t  # not accounting for scr refresh
                black_com.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_com.stopped')
                black_com.setAutoDraw(False)
        
        # *reward_sub* updates
        if reward_sub.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_sub.frameNStart = frameN  # exact frame index
            reward_sub.tStart = t  # local t and not account for scr refresh
            reward_sub.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_sub, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reward_sub.started')
            reward_sub.setAutoDraw(True)
        if reward_sub.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_sub.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                reward_sub.tStop = t  # not accounting for scr refresh
                reward_sub.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_sub.stopped')
                reward_sub.setAutoDraw(False)
        
        # *palyerb* updates
        if palyerb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            palyerb.frameNStart = frameN  # exact frame index
            palyerb.tStart = t  # local t and not account for scr refresh
            palyerb.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(palyerb, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'palyerb.started')
            palyerb.setAutoDraw(True)
        if palyerb.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > palyerb.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                palyerb.tStop = t  # not accounting for scr refresh
                palyerb.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'palyerb.stopped')
                palyerb.setAutoDraw(False)
        
        # *reward__sub* updates
        if reward__sub.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward__sub.frameNStart = frameN  # exact frame index
            reward__sub.tStart = t  # local t and not account for scr refresh
            reward__sub.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward__sub, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reward__sub.started')
            reward__sub.setAutoDraw(True)
        if reward__sub.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward__sub.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                reward__sub.tStop = t  # not accounting for scr refresh
                reward__sub.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward__sub.stopped')
                reward__sub.setAutoDraw(False)
        
        # *compare* updates
        if compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            compare.frameNStart = frameN  # exact frame index
            compare.tStart = t  # local t and not account for scr refresh
            compare.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(compare, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'compare.started')
            compare.setAutoDraw(True)
        if compare.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > compare.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                compare.tStop = t  # not accounting for scr refresh
                compare.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'compare.stopped')
                compare.setAutoDraw(False)
        
        # *red_compare* updates
        if red_compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            red_compare.frameNStart = frameN  # exact frame index
            red_compare.tStart = t  # local t and not account for scr refresh
            red_compare.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(red_compare, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'red_compare.started')
            red_compare.setAutoDraw(True)
        if red_compare.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > red_compare.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                red_compare.tStop = t  # not accounting for scr refresh
                red_compare.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_compare.stopped')
                red_compare.setAutoDraw(False)
        
        # *green_com* updates
        if green_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green_com.frameNStart = frameN  # exact frame index
            green_com.tStart = t  # local t and not account for scr refresh
            green_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green_com, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'green_com.started')
            green_com.setAutoDraw(True)
        if green_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > green_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                green_com.tStop = t  # not accounting for scr refresh
                green_com.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_com.stopped')
                green_com.setAutoDraw(False)
        
        # *yellow_com* updates
        if yellow_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            yellow_com.frameNStart = frameN  # exact frame index
            yellow_com.tStart = t  # local t and not account for scr refresh
            yellow_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellow_com, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'yellow_com.started')
            yellow_com.setAutoDraw(True)
        if yellow_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > yellow_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                yellow_com.tStop = t  # not accounting for scr refresh
                yellow_com.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow_com.stopped')
                yellow_com.setAutoDraw(False)
        
        # *blue_com* updates
        if blue_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blue_com.frameNStart = frameN  # exact frame index
            blue_com.tStart = t  # local t and not account for scr refresh
            blue_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blue_com, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blue_com.started')
            blue_com.setAutoDraw(True)
        if blue_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blue_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                blue_com.tStop = t  # not accounting for scr refresh
                blue_com.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_com.stopped')
                blue_com.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in comparison_resultComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "comparison_result" ---
    for thisComponent in comparison_resultComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.500000)
    
    # --- Prepare to start Routine "happy_rating" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from randomwalk
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        continueRoutine = True
    else:
        continueRoutine = False
    uyestart = uyellow
    uredstart = ured
    ubluestart = ublue
    ugreenstart = ugreen
    B_start[0] = B_0
    B_start[1] = B_1
    B_start[2] = B_2
    B_start[3] = B_3
    slider.reset()
    # keep track of which components have finished
    happy_ratingComponents = [image_3, slider, text, image_4, image_5]
    for thisComponent in happy_ratingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "happy_rating" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_3* updates
        if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_3.frameNStart = frameN  # exact frame index
            image_3.tStart = t  # local t and not account for scr refresh
            image_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_3.started')
            image_3.setAutoDraw(True)
        
        # *slider* updates
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider.started')
            slider.setAutoDraw(True)
        
        # Check slider for response to end routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            text.setAutoDraw(True)
        
        # *image_4* updates
        if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_4.frameNStart = frameN  # exact frame index
            image_4.tStart = t  # local t and not account for scr refresh
            image_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_4.started')
            image_4.setAutoDraw(True)
        
        # *image_5* updates
        if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_5.frameNStart = frameN  # exact frame index
            image_5.tStart = t  # local t and not account for scr refresh
            image_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_5.started')
            image_5.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in happy_ratingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "happy_rating" ---
    for thisComponent in happy_ratingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Run 'End Routine' code from randomwalk
    thisExp.addData('rewardyellow', str(rewardyellowlist[0]))
    thisExp.addData('rewardred', str(rewardredlist[0]))
    thisExp.addData('rewardblue', str(rewardbluelist[0]))
    thisExp.addData('rewardgreen', str(rewardgreenlist[0]))
    thisExp.addData('subchoose', str(alist[0]))
    thisExp.addData('playerb', str(jlist[0]))
    trials.addData('slider.response', slider.getRating())
    trials.addData('slider.rt', slider.getRT())
    # the Routine "happy_rating" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "expect" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    expect_self_black.setPos((black_x_pos, black_y_pos))
    expect_self_black.setImage('pic/black.png')
    expectself.reset()
    # Run 'Begin Routine' code from expect_self_2
    if (choose_bandit.keys == 'r' or choose_bandit.keys == 'f' or choose_bandit.keys == 'i' or choose_bandit.keys == 'j') and(expect_which==0):
        continueRoutine = True
    else:
        continueRoutine = False
    
    expect_self_red.setImage('pic/red.png')
    expect_self_green.setImage('pic/green.png')
    expect_self_yellow.setImage('pic/yellow.png')
    expect_self_blue.setImage('pic/blue.png')
    # setup some python lists for storing info about the tijiao_click
    tijiao_click.x = []
    tijiao_click.y = []
    tijiao_click.leftButton = []
    tijiao_click.midButton = []
    tijiao_click.rightButton = []
    tijiao_click.time = []
    tijiao_click.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    expectComponents = [expect_self_black, expect_self, expectself, expect_self_red, expect_self_green, expect_self_yellow, expect_self_blue, tijiao, tijiao_click]
    for thisComponent in expectComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "expect" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *expect_self_black* updates
        if expect_self_black.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            expect_self_black.frameNStart = frameN  # exact frame index
            expect_self_black.tStart = t  # local t and not account for scr refresh
            expect_self_black.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_self_black, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_self_black.started')
            expect_self_black.setAutoDraw(True)
        if expect_self_black.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_self_black.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_self_black.tStop = t  # not accounting for scr refresh
                expect_self_black.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_self_black.stopped')
                expect_self_black.setAutoDraw(False)
        
        # *expect_self* updates
        if expect_self.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_self.frameNStart = frameN  # exact frame index
            expect_self.tStart = t  # local t and not account for scr refresh
            expect_self.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_self, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_self.started')
            expect_self.setAutoDraw(True)
        if expect_self.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_self.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_self.tStop = t  # not accounting for scr refresh
                expect_self.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_self.stopped')
                expect_self.setAutoDraw(False)
        
        # *expectself* updates
        if expectself.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expectself.frameNStart = frameN  # exact frame index
            expectself.tStart = t  # local t and not account for scr refresh
            expectself.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expectself, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expectself.started')
            expectself.setAutoDraw(True)
        if expectself.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expectself.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expectself.tStop = t  # not accounting for scr refresh
                expectself.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expectself.stopped')
                expectself.setAutoDraw(False)
        
        # *expect_self_red* updates
        if expect_self_red.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_self_red.frameNStart = frameN  # exact frame index
            expect_self_red.tStart = t  # local t and not account for scr refresh
            expect_self_red.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_self_red, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_self_red.started')
            expect_self_red.setAutoDraw(True)
        if expect_self_red.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_self_red.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_self_red.tStop = t  # not accounting for scr refresh
                expect_self_red.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_self_red.stopped')
                expect_self_red.setAutoDraw(False)
        
        # *expect_self_green* updates
        if expect_self_green.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_self_green.frameNStart = frameN  # exact frame index
            expect_self_green.tStart = t  # local t and not account for scr refresh
            expect_self_green.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_self_green, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_self_green.started')
            expect_self_green.setAutoDraw(True)
        if expect_self_green.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_self_green.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_self_green.tStop = t  # not accounting for scr refresh
                expect_self_green.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_self_green.stopped')
                expect_self_green.setAutoDraw(False)
        
        # *expect_self_yellow* updates
        if expect_self_yellow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_self_yellow.frameNStart = frameN  # exact frame index
            expect_self_yellow.tStart = t  # local t and not account for scr refresh
            expect_self_yellow.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_self_yellow, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_self_yellow.started')
            expect_self_yellow.setAutoDraw(True)
        if expect_self_yellow.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_self_yellow.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_self_yellow.tStop = t  # not accounting for scr refresh
                expect_self_yellow.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_self_yellow.stopped')
                expect_self_yellow.setAutoDraw(False)
        
        # *expect_self_blue* updates
        if expect_self_blue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_self_blue.frameNStart = frameN  # exact frame index
            expect_self_blue.tStart = t  # local t and not account for scr refresh
            expect_self_blue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_self_blue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_self_blue.started')
            expect_self_blue.setAutoDraw(True)
        if expect_self_blue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_self_blue.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_self_blue.tStop = t  # not accounting for scr refresh
                expect_self_blue.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_self_blue.stopped')
                expect_self_blue.setAutoDraw(False)
        
        # *tijiao* updates
        if tijiao.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tijiao.frameNStart = frameN  # exact frame index
            tijiao.tStart = t  # local t and not account for scr refresh
            tijiao.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tijiao, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tijiao.started')
            tijiao.setAutoDraw(True)
        if tijiao.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > tijiao.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                tijiao.tStop = t  # not accounting for scr refresh
                tijiao.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tijiao.stopped')
                tijiao.setAutoDraw(False)
        # *tijiao_click* updates
        if tijiao_click.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tijiao_click.frameNStart = frameN  # exact frame index
            tijiao_click.tStart = t  # local t and not account for scr refresh
            tijiao_click.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tijiao_click, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('tijiao_click.started', t)
            tijiao_click.status = STARTED
            tijiao_click.mouseClock.reset()
            prevButtonState = tijiao_click.getPressed()  # if button is down already this ISN'T a new click
        if tijiao_click.status == STARTED:  # only update if started and not finished!
            buttons = tijiao_click.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    try:
                        iter(tijiao)
                        clickableList = tijiao
                    except:
                        clickableList = [tijiao]
                    for obj in clickableList:
                        if obj.contains(tijiao_click):
                            gotValidClick = True
                            tijiao_click.clicked_name.append(obj.name)
                    x, y = tijiao_click.getPos()
                    tijiao_click.x.append(x)
                    tijiao_click.y.append(y)
                    buttons = tijiao_click.getPressed()
                    tijiao_click.leftButton.append(buttons[0])
                    tijiao_click.midButton.append(buttons[1])
                    tijiao_click.rightButton.append(buttons[2])
                    tijiao_click.time.append(tijiao_click.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # abort routine on response
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in expectComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "expect" ---
    for thisComponent in expectComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('expectself.text',expectself.text)
    # Run 'End Routine' code from expect_self_2
    thisExp.addData('shuru',expectself.text)
    # store data for trials (TrialHandler)
    trials.addData('tijiao_click.x', tijiao_click.x)
    trials.addData('tijiao_click.y', tijiao_click.y)
    trials.addData('tijiao_click.leftButton', tijiao_click.leftButton)
    trials.addData('tijiao_click.midButton', tijiao_click.midButton)
    trials.addData('tijiao_click.rightButton', tijiao_click.rightButton)
    trials.addData('tijiao_click.time', tijiao_click.time)
    trials.addData('tijiao_click.clicked_name', tijiao_click.clicked_name)
    # the Routine "expect" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "expect_random" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    jiantou_random.setPos((black_x_random, 0.2))
    jiantou_random.setImage('pic/jiantou.png')
    expect_red.setImage('pic/red.png')
    expect_green.setImage('pic/green.png')
    expect_yellow.setImage('pic/yellow.png')
    expect_blue.setImage('pic/blue.png')
    # Run 'Begin Routine' code from code_5
    if (choose_bandit.keys == 'r' or choose_bandit.keys == 'f' or choose_bandit.keys == 'i' or choose_bandit.keys == 'j') and(expect_which==1):
        continueRoutine = True
    else:
        continueRoutine = False
    #if choose_bandit.keys == 'r':
    #    black_x_pos = -0.3
    #    black_y_pos = 0.225
    #elif choose_bandit.keys == 'f':
    #    black_x_pos = -0.3
    #    black_y_pos = -0.225
    #elif choose_bandit.keys == 'i':
    #    black_x_pos = 0.3
    #    black_y_pos = 0.225
    #elif choose_bandit.keys == 'j':
    #    black_x_pos = 0.3
    #    black_y_pos = -0.225
    #else:
    #    black_x_pos = 0
    #    black_y_pos = 0
    #    points = None
    #    continueRoutine = False
    expect_random_3.reset()
    # setup some python lists for storing info about the tijiao_random
    tijiao_random.x = []
    tijiao_random.y = []
    tijiao_random.leftButton = []
    tijiao_random.midButton = []
    tijiao_random.rightButton = []
    tijiao_random.time = []
    tijiao_random.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    expect_randomComponents = [jiantou_random, expect_red, expect_green, expect_yellow, expect_blue, expect_random_2, expect_random_3, tijiao_random, tijiao_random_2]
    for thisComponent in expect_randomComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "expect_random" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *jiantou_random* updates
        if jiantou_random.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            jiantou_random.frameNStart = frameN  # exact frame index
            jiantou_random.tStart = t  # local t and not account for scr refresh
            jiantou_random.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(jiantou_random, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'jiantou_random.started')
            jiantou_random.setAutoDraw(True)
        if jiantou_random.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > jiantou_random.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                jiantou_random.tStop = t  # not accounting for scr refresh
                jiantou_random.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'jiantou_random.stopped')
                jiantou_random.setAutoDraw(False)
        
        # *expect_red* updates
        if expect_red.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_red.frameNStart = frameN  # exact frame index
            expect_red.tStart = t  # local t and not account for scr refresh
            expect_red.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_red, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_red.started')
            expect_red.setAutoDraw(True)
        if expect_red.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_red.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_red.tStop = t  # not accounting for scr refresh
                expect_red.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_red.stopped')
                expect_red.setAutoDraw(False)
        
        # *expect_green* updates
        if expect_green.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_green.frameNStart = frameN  # exact frame index
            expect_green.tStart = t  # local t and not account for scr refresh
            expect_green.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_green, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_green.started')
            expect_green.setAutoDraw(True)
        if expect_green.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_green.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_green.tStop = t  # not accounting for scr refresh
                expect_green.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_green.stopped')
                expect_green.setAutoDraw(False)
        
        # *expect_yellow* updates
        if expect_yellow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_yellow.frameNStart = frameN  # exact frame index
            expect_yellow.tStart = t  # local t and not account for scr refresh
            expect_yellow.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_yellow, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_yellow.started')
            expect_yellow.setAutoDraw(True)
        if expect_yellow.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_yellow.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_yellow.tStop = t  # not accounting for scr refresh
                expect_yellow.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_yellow.stopped')
                expect_yellow.setAutoDraw(False)
        
        # *expect_blue* updates
        if expect_blue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_blue.frameNStart = frameN  # exact frame index
            expect_blue.tStart = t  # local t and not account for scr refresh
            expect_blue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_blue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_blue.started')
            expect_blue.setAutoDraw(True)
        if expect_blue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_blue.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_blue.tStop = t  # not accounting for scr refresh
                expect_blue.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_blue.stopped')
                expect_blue.setAutoDraw(False)
        
        # *expect_random_2* updates
        if expect_random_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_random_2.frameNStart = frameN  # exact frame index
            expect_random_2.tStart = t  # local t and not account for scr refresh
            expect_random_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_random_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_random_2.started')
            expect_random_2.setAutoDraw(True)
        if expect_random_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_random_2.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_random_2.tStop = t  # not accounting for scr refresh
                expect_random_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_random_2.stopped')
                expect_random_2.setAutoDraw(False)
        
        # *expect_random_3* updates
        if expect_random_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expect_random_3.frameNStart = frameN  # exact frame index
            expect_random_3.tStart = t  # local t and not account for scr refresh
            expect_random_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expect_random_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expect_random_3.started')
            expect_random_3.setAutoDraw(True)
        if expect_random_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expect_random_3.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expect_random_3.tStop = t  # not accounting for scr refresh
                expect_random_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expect_random_3.stopped')
                expect_random_3.setAutoDraw(False)
        # *tijiao_random* updates
        if tijiao_random.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tijiao_random.frameNStart = frameN  # exact frame index
            tijiao_random.tStart = t  # local t and not account for scr refresh
            tijiao_random.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tijiao_random, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('tijiao_random.started', t)
            tijiao_random.status = STARTED
            tijiao_random.mouseClock.reset()
            prevButtonState = tijiao_random.getPressed()  # if button is down already this ISN'T a new click
        if tijiao_random.status == STARTED:  # only update if started and not finished!
            buttons = tijiao_random.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    try:
                        iter(tijiao)
                        clickableList = tijiao
                    except:
                        clickableList = [tijiao]
                    for obj in clickableList:
                        if obj.contains(tijiao_random):
                            gotValidClick = True
                            tijiao_random.clicked_name.append(obj.name)
                    x, y = tijiao_random.getPos()
                    tijiao_random.x.append(x)
                    tijiao_random.y.append(y)
                    buttons = tijiao_random.getPressed()
                    tijiao_random.leftButton.append(buttons[0])
                    tijiao_random.midButton.append(buttons[1])
                    tijiao_random.rightButton.append(buttons[2])
                    tijiao_random.time.append(tijiao_random.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # abort routine on response
        
        # *tijiao_random_2* updates
        if tijiao_random_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tijiao_random_2.frameNStart = frameN  # exact frame index
            tijiao_random_2.tStart = t  # local t and not account for scr refresh
            tijiao_random_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tijiao_random_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tijiao_random_2.started')
            tijiao_random_2.setAutoDraw(True)
        if tijiao_random_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > tijiao_random_2.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                tijiao_random_2.tStop = t  # not accounting for scr refresh
                tijiao_random_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tijiao_random_2.stopped')
                tijiao_random_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in expect_randomComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "expect_random" ---
    for thisComponent in expect_randomComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('expect_random_3.text',expect_random_3.text)
    # store data for trials (TrialHandler)
    trials.addData('tijiao_random.x', tijiao_random.x)
    trials.addData('tijiao_random.y', tijiao_random.y)
    trials.addData('tijiao_random.leftButton', tijiao_random.leftButton)
    trials.addData('tijiao_random.midButton', tijiao_random.midButton)
    trials.addData('tijiao_random.rightButton', tijiao_random.rightButton)
    trials.addData('tijiao_random.time', tijiao_random.time)
    trials.addData('tijiao_random.clicked_name', tijiao_random.clicked_name)
    # the Routine "expect_random" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "confidence" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    slider_2.reset()
    # Run 'Begin Routine' code from code_3
    if (choose_bandit.keys == 'r' or choose_bandit.keys == 'f' or choose_bandit.keys == 'i' or choose_bandit.keys == 'j') :#and(expect_which==0)
        continueRoutine = True
    else:
        continueRoutine = False
    
    # keep track of which components have finished
    confidenceComponents = [slider_2, image_7, confi, notconfi]
    for thisComponent in confidenceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "confidence" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider_2* updates
        if slider_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_2.frameNStart = frameN  # exact frame index
            slider_2.tStart = t  # local t and not account for scr refresh
            slider_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_2.started')
            slider_2.setAutoDraw(True)
        
        # Check slider_2 for response to end routine
        if slider_2.getRating() is not None and slider_2.status == STARTED:
            continueRoutine = False
        
        # *image_7* updates
        if image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_7.frameNStart = frameN  # exact frame index
            image_7.tStart = t  # local t and not account for scr refresh
            image_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_7.started')
            image_7.setAutoDraw(True)
        
        # *confi* updates
        if confi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            confi.frameNStart = frameN  # exact frame index
            confi.tStart = t  # local t and not account for scr refresh
            confi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(confi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'confi.started')
            confi.setAutoDraw(True)
        
        # *notconfi* updates
        if notconfi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            notconfi.frameNStart = frameN  # exact frame index
            notconfi.tStart = t  # local t and not account for scr refresh
            notconfi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(notconfi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'notconfi.started')
            notconfi.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in confidenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "confidence" ---
    for thisComponent in confidenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('slider_2.response', slider_2.getRating())
    trials.addData('slider_2.rt', slider_2.getRT())
    # the Routine "confidence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "expectB" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    expectB_2.reset()
    # Run 'Begin Routine' code from expectb
    if (choose_bandit.keys == 'r' or choose_bandit.keys == 'f' or choose_bandit.keys == 'i' or choose_bandit.keys == 'j'): #and(expect_which==0)
        continueRoutine = True
    else:
        continueRoutine = False
    thisExp.addData('random_whicharm', black_x_random)
    # setup some python lists for storing info about the tijiao_B
    tijiao_B.x = []
    tijiao_B.y = []
    tijiao_B.leftButton = []
    tijiao_B.midButton = []
    tijiao_B.rightButton = []
    tijiao_B.time = []
    tijiao_B.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    expectBComponents = [image_8, expectB_2, tijiao_B_2, tijiao_B]
    for thisComponent in expectBComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "expectB" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_8* updates
        if image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_8.frameNStart = frameN  # exact frame index
            image_8.tStart = t  # local t and not account for scr refresh
            image_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_8.started')
            image_8.setAutoDraw(True)
        if image_8.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_8.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                image_8.tStop = t  # not accounting for scr refresh
                image_8.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_8.stopped')
                image_8.setAutoDraw(False)
        
        # *expectB_2* updates
        if expectB_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            expectB_2.frameNStart = frameN  # exact frame index
            expectB_2.tStart = t  # local t and not account for scr refresh
            expectB_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(expectB_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'expectB_2.started')
            expectB_2.setAutoDraw(True)
        if expectB_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > expectB_2.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                expectB_2.tStop = t  # not accounting for scr refresh
                expectB_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'expectB_2.stopped')
                expectB_2.setAutoDraw(False)
        
        # *tijiao_B_2* updates
        if tijiao_B_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tijiao_B_2.frameNStart = frameN  # exact frame index
            tijiao_B_2.tStart = t  # local t and not account for scr refresh
            tijiao_B_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tijiao_B_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tijiao_B_2.started')
            tijiao_B_2.setAutoDraw(True)
        if tijiao_B_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > tijiao_B_2.tStartRefresh + ss-frameTolerance:
                # keep track of stop time/frame for later
                tijiao_B_2.tStop = t  # not accounting for scr refresh
                tijiao_B_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tijiao_B_2.stopped')
                tijiao_B_2.setAutoDraw(False)
        # *tijiao_B* updates
        if tijiao_B.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tijiao_B.frameNStart = frameN  # exact frame index
            tijiao_B.tStart = t  # local t and not account for scr refresh
            tijiao_B.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tijiao_B, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('tijiao_B.started', t)
            tijiao_B.status = STARTED
            tijiao_B.mouseClock.reset()
            prevButtonState = tijiao_B.getPressed()  # if button is down already this ISN'T a new click
        if tijiao_B.status == STARTED:  # only update if started and not finished!
            buttons = tijiao_B.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    try:
                        iter(tijiao)
                        clickableList = tijiao
                    except:
                        clickableList = [tijiao]
                    for obj in clickableList:
                        if obj.contains(tijiao_B):
                            gotValidClick = True
                            tijiao_B.clicked_name.append(obj.name)
                    x, y = tijiao_B.getPos()
                    tijiao_B.x.append(x)
                    tijiao_B.y.append(y)
                    buttons = tijiao_B.getPressed()
                    tijiao_B.leftButton.append(buttons[0])
                    tijiao_B.midButton.append(buttons[1])
                    tijiao_B.rightButton.append(buttons[2])
                    tijiao_B.time.append(tijiao_B.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # abort routine on response
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in expectBComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "expectB" ---
    for thisComponent in expectBComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('expectB_2.text',expectB_2.text)
    # Run 'End Routine' code from expectb
    thisExp.addData('shuru',expectB_2.text)
    # store data for trials (TrialHandler)
    trials.addData('tijiao_B.x', tijiao_B.x)
    trials.addData('tijiao_B.y', tijiao_B.y)
    trials.addData('tijiao_B.leftButton', tijiao_B.leftButton)
    trials.addData('tijiao_B.midButton', tijiao_B.midButton)
    trials.addData('tijiao_B.rightButton', tijiao_B.rightButton)
    trials.addData('tijiao_B.time', tijiao_B.time)
    trials.addData('tijiao_B.clicked_name', tijiao_B.clicked_name)
    # the Routine "expectB" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "wrongsign" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        i=0
        continueRoutine = False
    else:
        i=6
        continueRoutine = True
    bigx.setText('×')
    # keep track of which components have finished
    wrongsignComponents = [bigx]
    for thisComponent in wrongsignComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "wrongsign" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *bigx* updates
        if bigx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            bigx.frameNStart = frameN  # exact frame index
            bigx.tStart = t  # local t and not account for scr refresh
            bigx.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bigx, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bigx.started')
            bigx.setAutoDraw(True)
        if bigx.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > bigx.tStartRefresh + i-frameTolerance:
                # keep track of stop time/frame for later
                bigx.tStop = t  # not accounting for scr refresh
                bigx.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bigx.stopped')
                bigx.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in wrongsignComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "wrongsign" ---
    for thisComponent in wrongsignComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "wrongsign" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intertrial" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_4
    import numpy as np
    intertrialtime = np.random.normal(2, 1)
    while (intertrialtime < 1.5) | (intertrialtime > 2):
        intertrialtime = np.random.normal(2,1)
    # keep track of which components have finished
    intertrialComponents = [intertrialfix]
    for thisComponent in intertrialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intertrial" ---
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intertrialfix* updates
        if intertrialfix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intertrialfix.frameNStart = frameN  # exact frame index
            intertrialfix.tStart = t  # local t and not account for scr refresh
            intertrialfix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intertrialfix, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intertrialfix.started')
            intertrialfix.setAutoDraw(True)
        if intertrialfix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > intertrialfix.tStartRefresh + intertrialtime-frameTolerance:
                # keep track of stop time/frame for later
                intertrialfix.tStop = t  # not accounting for scr refresh
                intertrialfix.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'intertrialfix.stopped')
                intertrialfix.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intertrialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intertrial" ---
    for thisComponent in intertrialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "intertrial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 150 repeats of 'trials'


# --- Prepare to start Routine "end" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
key_resp.keys = []
key_resp.rt = []
_key_resp_allKeys = []
# keep track of which components have finished
endComponents = [image, key_resp]
for thisComponent in endComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "end" ---
while continueRoutine and routineTimer.getTime() < 30.0:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *image* updates
    if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        image.frameNStart = frameN  # exact frame index
        image.tStart = t  # local t and not account for scr refresh
        image.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'image.started')
        image.setAutoDraw(True)
    if image.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > image.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            image.tStop = t  # not accounting for scr refresh
            image.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.stopped')
            image.setAutoDraw(False)
    
    # *key_resp* updates
    waitOnFlip = False
    if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp.frameNStart = frameN  # exact frame index
        key_resp.tStart = t  # local t and not account for scr refresh
        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'key_resp.started')
        key_resp.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > key_resp.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            key_resp.tStop = t  # not accounting for scr refresh
            key_resp.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.stopped')
            key_resp.status = FINISHED
    if key_resp.status == STARTED and not waitOnFlip:
        theseKeys = key_resp.getKeys(keyList=['y','n','left','right','space'], waitRelease=False)
        _key_resp_allKeys.extend(theseKeys)
        if len(_key_resp_allKeys):
            key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
            key_resp.rt = _key_resp_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "end" ---
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp.keys in ['', [], None]:  # No response was made
    key_resp.keys = None
thisExp.addData('key_resp.keys',key_resp.keys)
if key_resp.keys != None:  # we had a response
    thisExp.addData('key_resp.rt', key_resp.rt)
thisExp.nextEntry()
# using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
if routineForceEnded:
    routineTimer.reset()
else:
    routineTimer.addTime(-30.000000)

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
