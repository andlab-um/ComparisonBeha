#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on December 13, 2022, at 17:44
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

trialsN = 0
import numpy as np
import random
#vnml=np.random.normal(loc=0,scale=2.8,size=150)
#vzhengshu=np.round(vnml)#生成噪声v的高斯分布数列并四舍五入取整（same with原论文
#vshuzi=vzhengshu.tolist()
#v1=random.sample(vshuzi,1)
#v=v1[0]
import random

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
psychopyVersion = '2021.2.3'
expName = 'RL'  # from the Builder filename that created this script
expInfo = {'participant': '1', 'Gender': 'f', 'Age': '1'}
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
    originPath='C:\\Users\\user a\\Desktop\\RL_E1_final0\\RLe1final_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1280, 800], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instru"
instruClock = core.Clock()
instru_img1 = visual.ImageStim(
    win=win,
    name='instru_img1', units='height', 
    image='pic/instru1.png', mask=None,
    ori=0, pos=(0, 0), size=(1.56,0.8),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
instru_resp = keyboard.Keyboard()

# Initialize components for Routine "instr2"
instr2Clock = core.Clock()
instru_img2 = visual.ImageStim(
    win=win,
    name='instru_img2', 
    image='pic/instru2.png', mask=None,
    ori=0.0, pos=(0, 0), size=(1.56,0.9),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
instru_resp2 = keyboard.Keyboard()

# Initialize components for Routine "fixbefore"
fixbeforeClock = core.Clock()
fixbefore_2 = visual.TextStim(win=win, name='fixbefore_2',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "rest"
restClock = core.Clock()
rest_image = visual.ImageStim(
    win=win,
    name='rest_image', 
    image='pic/rest.png', mask=None,
    ori=0.0, pos=(0, 0), size=(0.857, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
rest_response = keyboard.Keyboard()

# Initialize components for Routine "choose"
chooseClock = core.Clock()
red = visual.ImageStim(
    win=win,
    name='red', 
    image='pic/red.png', mask=None,
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
green = visual.ImageStim(
    win=win,
    name='green', 
    image='pic/green.png', mask=None,
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
yellow = visual.ImageStim(
    win=win,
    name='yellow', 
    image='pic/yellow.png', mask=None,
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
blue = visual.ImageStim(
    win=win,
    name='blue', 
    image='pic/blue.png', mask=None,
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

# Initialize components for Routine "result"
resultClock = core.Clock()
black = visual.ImageStim(
    win=win,
    name='black', 
    image='sin', mask=None,
    ori=0, pos=[0,0], size=[0.33],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
red_2 = visual.ImageStim(
    win=win,
    name='red_2', 
    image='sin', mask=None,
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
green_2 = visual.ImageStim(
    win=win,
    name='green_2', 
    image='sin', mask=None,
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-3.0)
yellow_2 = visual.ImageStim(
    win=win,
    name='yellow_2', 
    image='sin', mask=None,
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
blue_2 = visual.ImageStim(
    win=win,
    name='blue_2', 
    image='sin', mask=None,
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
    image='sin', mask=None,
    ori=0.0, pos=(0, 0.3), size=(0.51, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)

# Initialize components for Routine "comparison_result"
comparison_resultClock = core.Clock()
black_com = visual.ImageStim(
    win=win,
    name='black_com', 
    image='pic/black.png', mask=None,
    ori=0, pos=[0,0], size=[0.33],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
reward_sub = visual.ImageStim(
    win=win,
    name='reward_sub', 
    image='pic/rewardsub.png', mask=None,
    ori=0.0, pos=(-0.215, 0.3), size=(0.365,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
palyerb = visual.ImageStim(
    win=win,
    name='palyerb', 
    image='pic/rewardpB.png', mask=None,
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
    image='pic/red.png', mask=None,
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-6.0)
green_com = visual.ImageStim(
    win=win,
    name='green_com', 
    image='pic/green.png', mask=None,
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-7.0)
yellow_com = visual.ImageStim(
    win=win,
    name='yellow_com', 
    image='pic/yellow.png', mask=None,
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-8.0)
blue_com = visual.ImageStim(
    win=win,
    name='blue_com', 
    image='pic/blue.png', mask=None,
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-9.0)

# Initialize components for Routine "slide"
slideClock = core.Clock()
image_3 = visual.ImageStim(
    win=win,
    name='image_3', 
    image='pic/happychoose.png', mask=None,
    ori=0.0, pos=(0, 0.2), size=(0.57, 0.1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
slider = visual.Slider(win=win, name='slider',
    startValue=None, size=(1.0, 0.1), pos=(0, 0), units=None,
    labels=None, ticks=(1,2,3,4,5,6,7), granularity=1.0,
    style='rating', styleTweaks=(), opacity=None,
    color='LightGray', fillColor='Red', borderColor='White', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-2, readOnly=False)
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
    image='pic/happy.png', mask=None,
    ori=0.0, pos=(0.7, 0), size=(0.183, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)
image_5 = visual.ImageStim(
    win=win,
    name='image_5', 
    image='pic/sad.png', mask=None,
    ori=0.0, pos=(-0.7, 0), size=(0.217, 0.2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-5.0)

# Initialize components for Routine "wrongsign"
wrongsignClock = core.Clock()
bigx = visual.TextStim(win=win, name='bigx',
    text='',
    font='Arial',
    pos=(0, 0), height=0.5, wrapWidth=None, ori=0, 
    color='red', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "intertrial"
intertrialClock = core.Clock()
intertrialfix = visual.TextStim(win=win, name='intertrialfix',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.06, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "end"
endClock = core.Clock()
image = visual.ImageStim(
    win=win,
    name='image', 
    image='pic/end.png', mask=None,
    ori=0, pos=(0, 0), size=(0.661,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
key_resp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instru"-------
continueRoutine = True
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
instruClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instru"-------
while continueRoutine:
    # get current time
    t = instruClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruClock)
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
        instru_img1.setAutoDraw(True)
    
    # *instru_resp* updates
    waitOnFlip = False
    if instru_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instru_resp.frameNStart = frameN  # exact frame index
        instru_resp.tStart = t  # local t and not account for scr refresh
        instru_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instru_resp, 'tStartRefresh')  # time at next scr refresh
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
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instru"-------
for thisComponent in instruComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instru_img1.started', instru_img1.tStartRefresh)
thisExp.addData('instru_img1.stopped', instru_img1.tStopRefresh)
# check responses
if instru_resp.keys in ['', [], None]:  # No response was made
    instru_resp.keys = None
thisExp.addData('instru_resp.keys',instru_resp.keys)
if instru_resp.keys != None:  # we had a response
    thisExp.addData('instru_resp.rt', instru_resp.rt)
thisExp.addData('instru_resp.started', instru_resp.tStartRefresh)
thisExp.addData('instru_resp.stopped', instru_resp.tStopRefresh)
thisExp.nextEntry()
# the Routine "instru" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr2"-------
continueRoutine = True
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
instr2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr2"-------
while continueRoutine:
    # get current time
    t = instr2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr2Clock)
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
        instru_img2.setAutoDraw(True)
    
    # *instru_resp2* updates
    waitOnFlip = False
    if instru_resp2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instru_resp2.frameNStart = frameN  # exact frame index
        instru_resp2.tStart = t  # local t and not account for scr refresh
        instru_resp2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instru_resp2, 'tStartRefresh')  # time at next scr refresh
        instru_resp2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instru_resp2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instru_resp2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instru_resp2.status == STARTED and not waitOnFlip:
        theseKeys = instru_resp2.getKeys(keyList=['space'], waitRelease=False)
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
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr2"-------
for thisComponent in instr2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('instru_img2.started', instru_img2.tStartRefresh)
thisExp.addData('instru_img2.stopped', instru_img2.tStopRefresh)
# check responses
if instru_resp2.keys in ['', [], None]:  # No response was made
    instru_resp2.keys = None
thisExp.addData('instru_resp2.keys',instru_resp2.keys)
if instru_resp2.keys != None:  # we had a response
    thisExp.addData('instru_resp2.rt', instru_resp2.rt)
thisExp.addData('instru_resp2.started', instru_resp2.tStartRefresh)
thisExp.addData('instru_resp2.stopped', instru_resp2.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "fixbefore"-------
continueRoutine = True
routineTimer.add(1.000000)
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
fixbeforeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "fixbefore"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = fixbeforeClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=fixbeforeClock)
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
        fixbefore_2.setAutoDraw(True)
    if fixbefore_2.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > fixbefore_2.tStartRefresh + 1-frameTolerance:
            # keep track of stop time/frame for later
            fixbefore_2.tStop = t  # not accounting for scr refresh
            fixbefore_2.frameNStop = frameN  # exact frame index
            win.timeOnFlip(fixbefore_2, 'tStopRefresh')  # time at next scr refresh
            fixbefore_2.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in fixbeforeComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "fixbefore"-------
for thisComponent in fixbeforeComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('fixbefore_2.started', fixbefore_2.tStartRefresh)
thisExp.addData('fixbefore_2.stopped', fixbefore_2.tStopRefresh)

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
    
    # ------Prepare to start Routine "rest"-------
    continueRoutine = True
    # update component parameters for each repeat
    rest_response.keys = []
    rest_response.rt = []
    _rest_response_allKeys = []
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
    restClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "rest"-------
    while continueRoutine:
        # get current time
        t = restClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=restClock)
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
            rest_image.setAutoDraw(True)
        
        # *rest_response* updates
        waitOnFlip = False
        if rest_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_response.frameNStart = frameN  # exact frame index
            rest_response.tStart = t  # local t and not account for scr refresh
            rest_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_response, 'tStartRefresh')  # time at next scr refresh
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
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in restComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "rest"-------
    for thisComponent in restComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('rest_image.started', rest_image.tStartRefresh)
    trials.addData('rest_image.stopped', rest_image.tStopRefresh)
    # check responses
    if rest_response.keys in ['', [], None]:  # No response was made
        rest_response.keys = None
    trials.addData('rest_response.keys',rest_response.keys)
    if rest_response.keys != None:  # we had a response
        trials.addData('rest_response.rt', rest_response.rt)
    trials.addData('rest_response.started', rest_response.tStartRefresh)
    trials.addData('rest_response.stopped', rest_response.tStopRefresh)
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "choose"-------
    continueRoutine = True
    routineTimer.add(1.500000)
    # update component parameters for each repeat
    choose_bandit.keys = []
    choose_bandit.rt = []
    _choose_bandit_allKeys = []
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
    chooseClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "choose"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = chooseClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=chooseClock)
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
            red.setAutoDraw(True)
        if red.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > red.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                red.tStop = t  # not accounting for scr refresh
                red.frameNStop = frameN  # exact frame index
                win.timeOnFlip(red, 'tStopRefresh')  # time at next scr refresh
                red.setAutoDraw(False)
        
        # *green* updates
        if green.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green.frameNStart = frameN  # exact frame index
            green.tStart = t  # local t and not account for scr refresh
            green.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green, 'tStartRefresh')  # time at next scr refresh
            green.setAutoDraw(True)
        if green.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > green.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                green.tStop = t  # not accounting for scr refresh
                green.frameNStop = frameN  # exact frame index
                win.timeOnFlip(green, 'tStopRefresh')  # time at next scr refresh
                green.setAutoDraw(False)
        
        # *yellow* updates
        if yellow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            yellow.frameNStart = frameN  # exact frame index
            yellow.tStart = t  # local t and not account for scr refresh
            yellow.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellow, 'tStartRefresh')  # time at next scr refresh
            yellow.setAutoDraw(True)
        if yellow.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > yellow.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                yellow.tStop = t  # not accounting for scr refresh
                yellow.frameNStop = frameN  # exact frame index
                win.timeOnFlip(yellow, 'tStopRefresh')  # time at next scr refresh
                yellow.setAutoDraw(False)
        
        # *blue* updates
        if blue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blue.frameNStart = frameN  # exact frame index
            blue.tStart = t  # local t and not account for scr refresh
            blue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blue, 'tStartRefresh')  # time at next scr refresh
            blue.setAutoDraw(True)
        if blue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blue.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                blue.tStop = t  # not accounting for scr refresh
                blue.frameNStop = frameN  # exact frame index
                win.timeOnFlip(blue, 'tStopRefresh')  # time at next scr refresh
                blue.setAutoDraw(False)
        
        # *choose_bandit* updates
        waitOnFlip = False
        if choose_bandit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choose_bandit.frameNStart = frameN  # exact frame index
            choose_bandit.tStart = t  # local t and not account for scr refresh
            choose_bandit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choose_bandit, 'tStartRefresh')  # time at next scr refresh
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
                win.timeOnFlip(choose_bandit, 'tStopRefresh')  # time at next scr refresh
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
            Fixation.setAutoDraw(True)
        if Fixation.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Fixation.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                Fixation.tStop = t  # not accounting for scr refresh
                Fixation.frameNStop = frameN  # exact frame index
                win.timeOnFlip(Fixation, 'tStopRefresh')  # time at next scr refresh
                Fixation.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in chooseComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "choose"-------
    for thisComponent in chooseComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('red.started', red.tStartRefresh)
    trials.addData('red.stopped', red.tStopRefresh)
    trials.addData('green.started', green.tStartRefresh)
    trials.addData('green.stopped', green.tStopRefresh)
    trials.addData('yellow.started', yellow.tStartRefresh)
    trials.addData('yellow.stopped', yellow.tStopRefresh)
    trials.addData('blue.started', blue.tStartRefresh)
    trials.addData('blue.stopped', blue.tStopRefresh)
    # check responses
    if choose_bandit.keys in ['', [], None]:  # No response was made
        choose_bandit.keys = None
    trials.addData('choose_bandit.keys',choose_bandit.keys)
    if choose_bandit.keys != None:  # we had a response
        trials.addData('choose_bandit.rt', choose_bandit.rt)
    trials.addData('choose_bandit.started', choose_bandit.tStartRefresh)
    trials.addData('choose_bandit.stopped', choose_bandit.tStopRefresh)
    trialsN = trialsN + 1
    trials.addData('Fixation.started', Fixation.tStartRefresh)
    trials.addData('Fixation.stopped', Fixation.tStopRefresh)
    
    # ------Prepare to start Routine "result"-------
    continueRoutine = True
    # update component parameters for each repeat
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
    elif choose_bandit.keys == 'f':
        black_x_pos = -0.2
        black_y_pos = 0
        docs_x_pos = -0.2
        docs_y_pos = 0
        points = int(np.round(rewardred))
    elif choose_bandit.keys == 'i':
        black_x_pos = 0.2
        black_y_pos = 0
        docs_x_pos = 0.2
        docs_y_pos = 0
        points = int(np.round(rewardblue))
    elif choose_bandit.keys == 'j':
        black_x_pos = 0.6
        black_y_pos = 0
        docs_x_pos = 0.6
        docs_y_pos = 0
        points = int(np.round(rewardgreen))
    else:
        black_x_pos = 9
        black_y_pos = 9
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
    resultComponents = [black, red_2, green_2, yellow_2, blue_2, fixation_2, docs, image_2]
    for thisComponent in resultComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    resultClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "result"-------
    while continueRoutine:
        # get current time
        t = resultClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=resultClock)
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
            black.setAutoDraw(True)
        if black.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > black.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                black.tStop = t  # not accounting for scr refresh
                black.frameNStop = frameN  # exact frame index
                win.timeOnFlip(black, 'tStopRefresh')  # time at next scr refresh
                black.setAutoDraw(False)
        
        # *red_2* updates
        if red_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            red_2.frameNStart = frameN  # exact frame index
            red_2.tStart = t  # local t and not account for scr refresh
            red_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(red_2, 'tStartRefresh')  # time at next scr refresh
            red_2.setAutoDraw(True)
        if red_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > red_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                red_2.tStop = t  # not accounting for scr refresh
                red_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(red_2, 'tStopRefresh')  # time at next scr refresh
                red_2.setAutoDraw(False)
        
        # *green_2* updates
        if green_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green_2.frameNStart = frameN  # exact frame index
            green_2.tStart = t  # local t and not account for scr refresh
            green_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green_2, 'tStartRefresh')  # time at next scr refresh
            green_2.setAutoDraw(True)
        if green_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > green_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                green_2.tStop = t  # not accounting for scr refresh
                green_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(green_2, 'tStopRefresh')  # time at next scr refresh
                green_2.setAutoDraw(False)
        
        # *yellow_2* updates
        if yellow_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            yellow_2.frameNStart = frameN  # exact frame index
            yellow_2.tStart = t  # local t and not account for scr refresh
            yellow_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellow_2, 'tStartRefresh')  # time at next scr refresh
            yellow_2.setAutoDraw(True)
        if yellow_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > yellow_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                yellow_2.tStop = t  # not accounting for scr refresh
                yellow_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(yellow_2, 'tStopRefresh')  # time at next scr refresh
                yellow_2.setAutoDraw(False)
        
        # *blue_2* updates
        if blue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blue_2.frameNStart = frameN  # exact frame index
            blue_2.tStart = t  # local t and not account for scr refresh
            blue_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blue_2, 'tStartRefresh')  # time at next scr refresh
            blue_2.setAutoDraw(True)
        if blue_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blue_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                blue_2.tStop = t  # not accounting for scr refresh
                blue_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(blue_2, 'tStopRefresh')  # time at next scr refresh
                blue_2.setAutoDraw(False)
        
        # *fixation_2* updates
        if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_2.frameNStart = frameN  # exact frame index
            fixation_2.tStart = t  # local t and not account for scr refresh
            fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
            fixation_2.setAutoDraw(True)
        if fixation_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_2.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                fixation_2.tStop = t  # not accounting for scr refresh
                fixation_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fixation_2, 'tStopRefresh')  # time at next scr refresh
                fixation_2.setAutoDraw(False)
        
        # *docs* updates
        if docs.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
            # keep track of start time/frame for later
            docs.frameNStart = frameN  # exact frame index
            docs.tStart = t  # local t and not account for scr refresh
            docs.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(docs, 'tStartRefresh')  # time at next scr refresh
            docs.setAutoDraw(True)
        if docs.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > docs.tStartRefresh + z-frameTolerance:
                # keep track of stop time/frame for later
                docs.tStop = t  # not accounting for scr refresh
                docs.frameNStop = frameN  # exact frame index
                win.timeOnFlip(docs, 'tStopRefresh')  # time at next scr refresh
                docs.setAutoDraw(False)
        
        # *image_2* updates
        if image_2.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            image_2.setAutoDraw(True)
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + z-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image_2, 'tStopRefresh')  # time at next scr refresh
                image_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resultComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "result"-------
    for thisComponent in resultComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('black.started', black.tStartRefresh)
    trials.addData('black.stopped', black.tStopRefresh)
    trials.addData('red_2.started', red_2.tStartRefresh)
    trials.addData('red_2.stopped', red_2.tStopRefresh)
    trials.addData('green_2.started', green_2.tStartRefresh)
    trials.addData('green_2.stopped', green_2.tStopRefresh)
    trials.addData('yellow_2.started', yellow_2.tStartRefresh)
    trials.addData('yellow_2.stopped', yellow_2.tStopRefresh)
    trials.addData('blue_2.started', blue_2.tStartRefresh)
    trials.addData('blue_2.stopped', blue_2.tStopRefresh)
    trials.addData('fixation_2.started', fixation_2.tStartRefresh)
    trials.addData('fixation_2.stopped', fixation_2.tStopRefresh)
    trials.addData('docs.started', docs.tStartRefresh)
    trials.addData('docs.stopped', docs.tStopRefresh)
    trials.addData('image_2.started', image_2.tStartRefresh)
    trials.addData('image_2.stopped', image_2.tStopRefresh)
    # the Routine "result" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "comparison_result"-------
    continueRoutine = True
    routineTimer.add(2.500000)
    # update component parameters for each repeat
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
    comparison_resultClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "comparison_result"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = comparison_resultClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=comparison_resultClock)
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
            black_com.setAutoDraw(True)
        if black_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > black_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                black_com.tStop = t  # not accounting for scr refresh
                black_com.frameNStop = frameN  # exact frame index
                win.timeOnFlip(black_com, 'tStopRefresh')  # time at next scr refresh
                black_com.setAutoDraw(False)
        
        # *reward_sub* updates
        if reward_sub.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_sub.frameNStart = frameN  # exact frame index
            reward_sub.tStart = t  # local t and not account for scr refresh
            reward_sub.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_sub, 'tStartRefresh')  # time at next scr refresh
            reward_sub.setAutoDraw(True)
        if reward_sub.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_sub.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                reward_sub.tStop = t  # not accounting for scr refresh
                reward_sub.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reward_sub, 'tStopRefresh')  # time at next scr refresh
                reward_sub.setAutoDraw(False)
        
        # *palyerb* updates
        if palyerb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            palyerb.frameNStart = frameN  # exact frame index
            palyerb.tStart = t  # local t and not account for scr refresh
            palyerb.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(palyerb, 'tStartRefresh')  # time at next scr refresh
            palyerb.setAutoDraw(True)
        if palyerb.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > palyerb.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                palyerb.tStop = t  # not accounting for scr refresh
                palyerb.frameNStop = frameN  # exact frame index
                win.timeOnFlip(palyerb, 'tStopRefresh')  # time at next scr refresh
                palyerb.setAutoDraw(False)
        
        # *reward__sub* updates
        if reward__sub.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward__sub.frameNStart = frameN  # exact frame index
            reward__sub.tStart = t  # local t and not account for scr refresh
            reward__sub.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward__sub, 'tStartRefresh')  # time at next scr refresh
            reward__sub.setAutoDraw(True)
        if reward__sub.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward__sub.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                reward__sub.tStop = t  # not accounting for scr refresh
                reward__sub.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reward__sub, 'tStopRefresh')  # time at next scr refresh
                reward__sub.setAutoDraw(False)
        
        # *compare* updates
        if compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            compare.frameNStart = frameN  # exact frame index
            compare.tStart = t  # local t and not account for scr refresh
            compare.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(compare, 'tStartRefresh')  # time at next scr refresh
            compare.setAutoDraw(True)
        if compare.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > compare.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                compare.tStop = t  # not accounting for scr refresh
                compare.frameNStop = frameN  # exact frame index
                win.timeOnFlip(compare, 'tStopRefresh')  # time at next scr refresh
                compare.setAutoDraw(False)
        
        # *red_compare* updates
        if red_compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            red_compare.frameNStart = frameN  # exact frame index
            red_compare.tStart = t  # local t and not account for scr refresh
            red_compare.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(red_compare, 'tStartRefresh')  # time at next scr refresh
            red_compare.setAutoDraw(True)
        if red_compare.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > red_compare.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                red_compare.tStop = t  # not accounting for scr refresh
                red_compare.frameNStop = frameN  # exact frame index
                win.timeOnFlip(red_compare, 'tStopRefresh')  # time at next scr refresh
                red_compare.setAutoDraw(False)
        
        # *green_com* updates
        if green_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            green_com.frameNStart = frameN  # exact frame index
            green_com.tStart = t  # local t and not account for scr refresh
            green_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(green_com, 'tStartRefresh')  # time at next scr refresh
            green_com.setAutoDraw(True)
        if green_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > green_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                green_com.tStop = t  # not accounting for scr refresh
                green_com.frameNStop = frameN  # exact frame index
                win.timeOnFlip(green_com, 'tStopRefresh')  # time at next scr refresh
                green_com.setAutoDraw(False)
        
        # *yellow_com* updates
        if yellow_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            yellow_com.frameNStart = frameN  # exact frame index
            yellow_com.tStart = t  # local t and not account for scr refresh
            yellow_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellow_com, 'tStartRefresh')  # time at next scr refresh
            yellow_com.setAutoDraw(True)
        if yellow_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > yellow_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                yellow_com.tStop = t  # not accounting for scr refresh
                yellow_com.frameNStop = frameN  # exact frame index
                win.timeOnFlip(yellow_com, 'tStopRefresh')  # time at next scr refresh
                yellow_com.setAutoDraw(False)
        
        # *blue_com* updates
        if blue_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blue_com.frameNStart = frameN  # exact frame index
            blue_com.tStart = t  # local t and not account for scr refresh
            blue_com.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blue_com, 'tStartRefresh')  # time at next scr refresh
            blue_com.setAutoDraw(True)
        if blue_com.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blue_com.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                blue_com.tStop = t  # not accounting for scr refresh
                blue_com.frameNStop = frameN  # exact frame index
                win.timeOnFlip(blue_com, 'tStopRefresh')  # time at next scr refresh
                blue_com.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in comparison_resultComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "comparison_result"-------
    for thisComponent in comparison_resultComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('black_com.started', black_com.tStartRefresh)
    trials.addData('black_com.stopped', black_com.tStopRefresh)
    trials.addData('reward_sub.started', reward_sub.tStartRefresh)
    trials.addData('reward_sub.stopped', reward_sub.tStopRefresh)
    trials.addData('palyerb.started', palyerb.tStartRefresh)
    trials.addData('palyerb.stopped', palyerb.tStopRefresh)
    trials.addData('reward__sub.started', reward__sub.tStartRefresh)
    trials.addData('reward__sub.stopped', reward__sub.tStopRefresh)
    trials.addData('compare.started', compare.tStartRefresh)
    trials.addData('compare.stopped', compare.tStopRefresh)
    trials.addData('red_compare.started', red_compare.tStartRefresh)
    trials.addData('red_compare.stopped', red_compare.tStopRefresh)
    trials.addData('green_com.started', green_com.tStartRefresh)
    trials.addData('green_com.stopped', green_com.tStopRefresh)
    trials.addData('yellow_com.started', yellow_com.tStartRefresh)
    trials.addData('yellow_com.stopped', yellow_com.tStopRefresh)
    trials.addData('blue_com.started', blue_com.tStartRefresh)
    trials.addData('blue_com.stopped', blue_com.tStopRefresh)
    
    # ------Prepare to start Routine "slide"-------
    continueRoutine = True
    # update component parameters for each repeat
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
    slideComponents = [image_3, slider, text, image_4, image_5]
    for thisComponent in slideComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    slideClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "slide"-------
    while continueRoutine:
        # get current time
        t = slideClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=slideClock)
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
            image_3.setAutoDraw(True)
        
        # *slider* updates
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
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
            text.setAutoDraw(True)
        
        # *image_4* updates
        if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_4.frameNStart = frameN  # exact frame index
            image_4.tStart = t  # local t and not account for scr refresh
            image_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
            image_4.setAutoDraw(True)
        
        # *image_5* updates
        if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_5.frameNStart = frameN  # exact frame index
            image_5.tStart = t  # local t and not account for scr refresh
            image_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
            image_5.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in slideComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "slide"-------
    for thisComponent in slideComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('rewardyellow', str(rewardyellowlist[0]))
    thisExp.addData('rewardred', str(rewardredlist[0]))
    thisExp.addData('rewardblue', str(rewardbluelist[0]))
    thisExp.addData('rewardgreen', str(rewardgreenlist[0]))
    thisExp.addData('subchoose', str(alist[0]))
    thisExp.addData('playerb', str(jlist[0]))
    trials.addData('image_3.started', image_3.tStartRefresh)
    trials.addData('image_3.stopped', image_3.tStopRefresh)
    trials.addData('slider.response', slider.getRating())
    trials.addData('slider.rt', slider.getRT())
    trials.addData('slider.started', slider.tStartRefresh)
    trials.addData('slider.stopped', slider.tStopRefresh)
    trials.addData('text.started', text.tStartRefresh)
    trials.addData('text.stopped', text.tStopRefresh)
    trials.addData('image_4.started', image_4.tStartRefresh)
    trials.addData('image_4.stopped', image_4.tStopRefresh)
    trials.addData('image_5.started', image_5.tStartRefresh)
    trials.addData('image_5.stopped', image_5.tStopRefresh)
    # the Routine "slide" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "wrongsign"-------
    continueRoutine = True
    # update component parameters for each repeat
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        i=0
        continueRoutine = False
    else:
        i=4.5
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
    wrongsignClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "wrongsign"-------
    while continueRoutine:
        # get current time
        t = wrongsignClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=wrongsignClock)
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
            bigx.setAutoDraw(True)
        if bigx.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > bigx.tStartRefresh + i-frameTolerance:
                # keep track of stop time/frame for later
                bigx.tStop = t  # not accounting for scr refresh
                bigx.frameNStop = frameN  # exact frame index
                win.timeOnFlip(bigx, 'tStopRefresh')  # time at next scr refresh
                bigx.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in wrongsignComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "wrongsign"-------
    for thisComponent in wrongsignComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('bigx.started', bigx.tStartRefresh)
    trials.addData('bigx.stopped', bigx.tStopRefresh)
    # the Routine "wrongsign" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "intertrial"-------
    continueRoutine = True
    # update component parameters for each repeat
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
    intertrialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "intertrial"-------
    while continueRoutine:
        # get current time
        t = intertrialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=intertrialClock)
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
            intertrialfix.setAutoDraw(True)
        if intertrialfix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > intertrialfix.tStartRefresh + intertrialtime-frameTolerance:
                # keep track of stop time/frame for later
                intertrialfix.tStop = t  # not accounting for scr refresh
                intertrialfix.frameNStop = frameN  # exact frame index
                win.timeOnFlip(intertrialfix, 'tStopRefresh')  # time at next scr refresh
                intertrialfix.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intertrialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "intertrial"-------
    for thisComponent in intertrialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('intertrialfix.started', intertrialfix.tStartRefresh)
    trials.addData('intertrialfix.stopped', intertrialfix.tStopRefresh)
    # the Routine "intertrial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 150 repeats of 'trials'


# ------Prepare to start Routine "end"-------
continueRoutine = True
routineTimer.add(30.000000)
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
endClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = endClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=endClock)
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
        image.setAutoDraw(True)
    if image.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > image.tStartRefresh + 30-frameTolerance:
            # keep track of stop time/frame for later
            image.tStop = t  # not accounting for scr refresh
            image.frameNStop = frameN  # exact frame index
            win.timeOnFlip(image, 'tStopRefresh')  # time at next scr refresh
            image.setAutoDraw(False)
    
    # *key_resp* updates
    waitOnFlip = False
    if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp.frameNStart = frameN  # exact frame index
        key_resp.tStart = t  # local t and not account for scr refresh
        key_resp.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
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
            win.timeOnFlip(key_resp, 'tStopRefresh')  # time at next scr refresh
            key_resp.status = FINISHED
    if key_resp.status == STARTED and not waitOnFlip:
        theseKeys = key_resp.getKeys(keyList=['y', 'n', 'left', 'right', 'space'], waitRelease=False)
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
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end"-------
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('image.started', image.tStartRefresh)
thisExp.addData('image.stopped', image.tStopRefresh)
# check responses
if key_resp.keys in ['', [], None]:  # No response was made
    key_resp.keys = None
thisExp.addData('key_resp.keys',key_resp.keys)
if key_resp.keys != None:  # we had a response
    thisExp.addData('key_resp.rt', key_resp.rt)
thisExp.addData('key_resp.started', key_resp.tStartRefresh)
thisExp.addData('key_resp.stopped', key_resp.tStopRefresh)
thisExp.nextEntry()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
