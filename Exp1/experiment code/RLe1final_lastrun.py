#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1post4),
    on 九月 03, 2024, at 14:04
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1post4'
expName = 'RL'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '1',
    'Gender': 'f',
    'Age': '1',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 800]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s' % (expInfo['participant'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='E:\\Comparison\\ComparisonBeha\\Exp1\\experiment code\\RLe1final_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('instru_resp') is None:
        # initialise instru_resp
        instru_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instru_resp',
        )
    if deviceManager.getDevice('instru_resp2') is None:
        # initialise instru_resp2
        instru_resp2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instru_resp2',
        )
    if deviceManager.getDevice('rest_response') is None:
        # initialise rest_response
        rest_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rest_response',
        )
    if deviceManager.getDevice('choose_bandit') is None:
        # initialise choose_bandit
        choose_bandit = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='choose_bandit',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instru" ---
    instru_img1 = visual.ImageStim(
        win=win,
        name='instru_img1', units='height', 
        image='pic/instru1.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(1.56,0.8),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=0.0)
    instru_resp = keyboard.Keyboard(deviceName='instru_resp')
    
    # --- Initialize components for Routine "instr2" ---
    instru_img2 = visual.ImageStim(
        win=win,
        name='instru_img2', 
        image='pic/instru2.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.56,0.9),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instru_resp2 = keyboard.Keyboard(deviceName='instru_resp2')
    
    # --- Initialize components for Routine "fixbefore" ---
    fixbefore_2 = visual.TextStim(win=win, name='fixbefore_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_image = visual.ImageStim(
        win=win,
        name='rest_image', 
        image='pic/rest.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.857, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    rest_response = keyboard.Keyboard(deviceName='rest_response')
    
    # --- Initialize components for Routine "choose" ---
    red = visual.ImageStim(
        win=win,
        name='red', 
        image='pic/red.png', mask=None, anchor='center',
        ori=0, pos=(-0.2, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=0.0)
    green = visual.ImageStim(
        win=win,
        name='green', 
        image='pic/green.png', mask=None, anchor='center',
        ori=0, pos=(0.6, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    yellow = visual.ImageStim(
        win=win,
        name='yellow', 
        image='pic/yellow.png', mask=None, anchor='center',
        ori=0, pos=(-0.6, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    blue = visual.ImageStim(
        win=win,
        name='blue', 
        image='pic/blue.png', mask=None, anchor='center',
        ori=0, pos=(0.2, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    choose_bandit = keyboard.Keyboard(deviceName='choose_bandit')
    Fixation = visual.TextStim(win=win, name='Fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    
    # --- Initialize components for Routine "result" ---
    black = visual.ImageStim(
        win=win,
        name='black', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0,0], draggable=False, size=[0.33],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    red_2 = visual.ImageStim(
        win=win,
        name='red_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(-0.2, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    green_2 = visual.ImageStim(
        win=win,
        name='green_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0.6, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-3.0)
    yellow_2 = visual.ImageStim(
        win=win,
        name='yellow_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(-0.6, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-4.0)
    blue_2 = visual.ImageStim(
        win=win,
        name='blue_2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=(0.2, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-5.0)
    fixation_2 = visual.TextStim(win=win, name='fixation_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-6.0);
    docs = visual.TextStim(win=win, name='docs',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-7.0);
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.51, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    
    # --- Initialize components for Routine "comparison_result" ---
    black_com = visual.ImageStim(
        win=win,
        name='black_com', 
        image='pic/black.png', mask=None, anchor='center',
        ori=0, pos=[0,0], draggable=False, size=[0.33],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-1.0)
    reward_sub = visual.ImageStim(
        win=win,
        name='reward_sub', 
        image='pic/rewardsub.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.215, 0.3), draggable=False, size=(0.365,0.1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    palyerb = visual.ImageStim(
        win=win,
        name='palyerb', 
        image='pic/rewardpB.png', mask=None, anchor='center',
        ori=0.0, pos=(0.215, 0.3), draggable=False, size=(0.427,0.1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    reward__sub = visual.TextStim(win=win, name='reward__sub',
        text='',
        font='Open Sans',
        pos=(-0.15, 0.305), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    compare = visual.TextStim(win=win, name='compare',
        text='',
        font='Open Sans',
        pos=(0.321, 0.305), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    red_compare = visual.ImageStim(
        win=win,
        name='red_compare', 
        image='pic/red.png', mask=None, anchor='center',
        ori=0, pos=(-0.2, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-6.0)
    green_com = visual.ImageStim(
        win=win,
        name='green_com', 
        image='pic/green.png', mask=None, anchor='center',
        ori=0, pos=(0.6, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-7.0)
    yellow_com = visual.ImageStim(
        win=win,
        name='yellow_com', 
        image='pic/yellow.png', mask=None, anchor='center',
        ori=0, pos=(-0.6, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-8.0)
    blue_com = visual.ImageStim(
        win=win,
        name='blue_com', 
        image='pic/blue.png', mask=None, anchor='center',
        ori=0, pos=(0.2, 0), draggable=False, size=[0.3],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-9.0)
    
    # --- Initialize components for Routine "slide" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='pic/happychoose.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.57, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    slider = visual.Slider(win=win, name='slider',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=None, ticks=(1,2,3,4,5,6,7), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text = visual.TextStim(win=win, name='text',
        text=None,
        font='Open Sans',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', 
        image='pic/happy.png', mask=None, anchor='center',
        ori=0.0, pos=(0.7, 0), draggable=False, size=(0.183, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', 
        image='pic/sad.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.7, 0), draggable=False, size=(0.217, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    
    # --- Initialize components for Routine "wrongsign" ---
    bigx = visual.TextStim(win=win, name='bigx',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0, 
        color='red', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "intertrial" ---
    intertrialfix = visual.TextStim(win=win, name='intertrialfix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='pic/end.png', mask=None, anchor='center',
        ori=0, pos=(0, 0), draggable=False, size=(0.661,0.1),
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=0.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instru" ---
    # create an object to store info about Routine instru
    instru = data.Routine(
        name='instru',
        components=[instru_img1, instru_resp],
    )
    instru.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instru_resp
    instru_resp.keys = []
    instru_resp.rt = []
    _instru_resp_allKeys = []
    # store start times for instru
    instru.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instru.tStart = globalClock.getTime(format='float')
    instru.status = STARTED
    thisExp.addData('instru.started', instru.tStart)
    instru.maxDuration = None
    # keep track of which components have finished
    instruComponents = instru.components
    for thisComponent in instru.components:
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
    instru.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instru_img1* updates
        
        # if instru_img1 is starting this frame...
        if instru_img1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instru_img1.frameNStart = frameN  # exact frame index
            instru_img1.tStart = t  # local t and not account for scr refresh
            instru_img1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instru_img1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instru_img1.started')
            # update status
            instru_img1.status = STARTED
            instru_img1.setAutoDraw(True)
        
        # if instru_img1 is active this frame...
        if instru_img1.status == STARTED:
            # update params
            pass
        
        # *instru_resp* updates
        waitOnFlip = False
        
        # if instru_resp is starting this frame...
        if instru_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instru_resp.frameNStart = frameN  # exact frame index
            instru_resp.tStart = t  # local t and not account for scr refresh
            instru_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instru_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instru_resp.started')
            # update status
            instru_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instru_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instru_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instru_resp.status == STARTED and not waitOnFlip:
            theseKeys = instru_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instru_resp_allKeys.extend(theseKeys)
            if len(_instru_resp_allKeys):
                instru_resp.keys = _instru_resp_allKeys[-1].name  # just the last key pressed
                instru_resp.rt = _instru_resp_allKeys[-1].rt
                instru_resp.duration = _instru_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instru.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instru.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instru" ---
    for thisComponent in instru.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instru
    instru.tStop = globalClock.getTime(format='float')
    instru.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instru.stopped', instru.tStop)
    # check responses
    if instru_resp.keys in ['', [], None]:  # No response was made
        instru_resp.keys = None
    thisExp.addData('instru_resp.keys',instru_resp.keys)
    if instru_resp.keys != None:  # we had a response
        thisExp.addData('instru_resp.rt', instru_resp.rt)
        thisExp.addData('instru_resp.duration', instru_resp.duration)
    thisExp.nextEntry()
    # the Routine "instru" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr2" ---
    # create an object to store info about Routine instr2
    instr2 = data.Routine(
        name='instr2',
        components=[instru_img2, instru_resp2],
    )
    instr2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instru_resp2
    instru_resp2.keys = []
    instru_resp2.rt = []
    _instru_resp2_allKeys = []
    # store start times for instr2
    instr2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr2.tStart = globalClock.getTime(format='float')
    instr2.status = STARTED
    thisExp.addData('instr2.started', instr2.tStart)
    instr2.maxDuration = None
    # keep track of which components have finished
    instr2Components = instr2.components
    for thisComponent in instr2.components:
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
    instr2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instru_img2* updates
        
        # if instru_img2 is starting this frame...
        if instru_img2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instru_img2.frameNStart = frameN  # exact frame index
            instru_img2.tStart = t  # local t and not account for scr refresh
            instru_img2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instru_img2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instru_img2.started')
            # update status
            instru_img2.status = STARTED
            instru_img2.setAutoDraw(True)
        
        # if instru_img2 is active this frame...
        if instru_img2.status == STARTED:
            # update params
            pass
        
        # *instru_resp2* updates
        waitOnFlip = False
        
        # if instru_resp2 is starting this frame...
        if instru_resp2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instru_resp2.frameNStart = frameN  # exact frame index
            instru_resp2.tStart = t  # local t and not account for scr refresh
            instru_resp2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instru_resp2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instru_resp2.started')
            # update status
            instru_resp2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instru_resp2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instru_resp2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instru_resp2.status == STARTED and not waitOnFlip:
            theseKeys = instru_resp2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instru_resp2_allKeys.extend(theseKeys)
            if len(_instru_resp2_allKeys):
                instru_resp2.keys = _instru_resp2_allKeys[-1].name  # just the last key pressed
                instru_resp2.rt = _instru_resp2_allKeys[-1].rt
                instru_resp2.duration = _instru_resp2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr2" ---
    for thisComponent in instr2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr2
    instr2.tStop = globalClock.getTime(format='float')
    instr2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr2.stopped', instr2.tStop)
    # check responses
    if instru_resp2.keys in ['', [], None]:  # No response was made
        instru_resp2.keys = None
    thisExp.addData('instru_resp2.keys',instru_resp2.keys)
    if instru_resp2.keys != None:  # we had a response
        thisExp.addData('instru_resp2.rt', instru_resp2.rt)
        thisExp.addData('instru_resp2.duration', instru_resp2.duration)
    thisExp.nextEntry()
    # the Routine "instr2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "fixbefore" ---
    # create an object to store info about Routine fixbefore
    fixbefore = data.Routine(
        name='fixbefore',
        components=[fixbefore_2],
    )
    fixbefore.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for fixbefore
    fixbefore.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    fixbefore.tStart = globalClock.getTime(format='float')
    fixbefore.status = STARTED
    thisExp.addData('fixbefore.started', fixbefore.tStart)
    fixbefore.maxDuration = None
    # keep track of which components have finished
    fixbeforeComponents = fixbefore.components
    for thisComponent in fixbefore.components:
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
    fixbefore.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixbefore_2* updates
        
        # if fixbefore_2 is starting this frame...
        if fixbefore_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixbefore_2.frameNStart = frameN  # exact frame index
            fixbefore_2.tStart = t  # local t and not account for scr refresh
            fixbefore_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixbefore_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixbefore_2.started')
            # update status
            fixbefore_2.status = STARTED
            fixbefore_2.setAutoDraw(True)
        
        # if fixbefore_2 is active this frame...
        if fixbefore_2.status == STARTED:
            # update params
            pass
        
        # if fixbefore_2 is stopping this frame...
        if fixbefore_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixbefore_2.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                fixbefore_2.tStop = t  # not accounting for scr refresh
                fixbefore_2.tStopRefresh = tThisFlipGlobal  # on global time
                fixbefore_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixbefore_2.stopped')
                # update status
                fixbefore_2.status = FINISHED
                fixbefore_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            fixbefore.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in fixbefore.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "fixbefore" ---
    for thisComponent in fixbefore.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for fixbefore
    fixbefore.tStop = globalClock.getTime(format='float')
    fixbefore.tStopRefresh = tThisFlipGlobal
    thisExp.addData('fixbefore.stopped', fixbefore.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if fixbefore.maxDurationReached:
        routineTimer.addTime(-fixbefore.maxDuration)
    elif fixbefore.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=150, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('sequence.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[rest_image, rest_response],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for rest_response
        rest_response.keys = []
        rest_response.rt = []
        _rest_response_allKeys = []
        # Run 'Begin Routine' code from code_rest
        if trialsN == 0 or trialsN % 75 != 0:
            continueRoutine = False
        
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
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
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_image* updates
            
            # if rest_image is starting this frame...
            if rest_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_image.frameNStart = frameN  # exact frame index
                rest_image.tStart = t  # local t and not account for scr refresh
                rest_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_image.started')
                # update status
                rest_image.status = STARTED
                rest_image.setAutoDraw(True)
            
            # if rest_image is active this frame...
            if rest_image.status == STARTED:
                # update params
                pass
            
            # *rest_response* updates
            waitOnFlip = False
            
            # if rest_response is starting this frame...
            if rest_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_response.frameNStart = frameN  # exact frame index
                rest_response.tStart = t  # local t and not account for scr refresh
                rest_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_response.started')
                # update status
                rest_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(rest_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(rest_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if rest_response.status == STARTED and not waitOnFlip:
                theseKeys = rest_response.getKeys(keyList=['q'], ignoreKeys=["escape"], waitRelease=False)
                _rest_response_allKeys.extend(theseKeys)
                if len(_rest_response_allKeys):
                    rest_response.keys = _rest_response_allKeys[-1].name  # just the last key pressed
                    rest_response.rt = _rest_response_allKeys[-1].rt
                    rest_response.duration = _rest_response_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # check responses
        if rest_response.keys in ['', [], None]:  # No response was made
            rest_response.keys = None
        trials.addData('rest_response.keys',rest_response.keys)
        if rest_response.keys != None:  # we had a response
            trials.addData('rest_response.rt', rest_response.rt)
            trials.addData('rest_response.duration', rest_response.duration)
        # the Routine "rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "choose" ---
        # create an object to store info about Routine choose
        choose = data.Routine(
            name='choose',
            components=[red, green, yellow, blue, choose_bandit, Fixation],
        )
        choose.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for choose_bandit
        choose_bandit.keys = []
        choose_bandit.rt = []
        _choose_bandit_allKeys = []
        # store start times for choose
        choose.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        choose.tStart = globalClock.getTime(format='float')
        choose.status = STARTED
        thisExp.addData('choose.started', choose.tStart)
        choose.maxDuration = None
        # keep track of which components have finished
        chooseComponents = choose.components
        for thisComponent in choose.components:
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
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        choose.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *red* updates
            
            # if red is starting this frame...
            if red.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                red.frameNStart = frameN  # exact frame index
                red.tStart = t  # local t and not account for scr refresh
                red.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red.started')
                # update status
                red.status = STARTED
                red.setAutoDraw(True)
            
            # if red is active this frame...
            if red.status == STARTED:
                # update params
                pass
            
            # if red is stopping this frame...
            if red.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > red.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    red.tStop = t  # not accounting for scr refresh
                    red.tStopRefresh = tThisFlipGlobal  # on global time
                    red.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red.stopped')
                    # update status
                    red.status = FINISHED
                    red.setAutoDraw(False)
            
            # *green* updates
            
            # if green is starting this frame...
            if green.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                green.frameNStart = frameN  # exact frame index
                green.tStart = t  # local t and not account for scr refresh
                green.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green.started')
                # update status
                green.status = STARTED
                green.setAutoDraw(True)
            
            # if green is active this frame...
            if green.status == STARTED:
                # update params
                pass
            
            # if green is stopping this frame...
            if green.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > green.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    green.tStop = t  # not accounting for scr refresh
                    green.tStopRefresh = tThisFlipGlobal  # on global time
                    green.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green.stopped')
                    # update status
                    green.status = FINISHED
                    green.setAutoDraw(False)
            
            # *yellow* updates
            
            # if yellow is starting this frame...
            if yellow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                yellow.frameNStart = frameN  # exact frame index
                yellow.tStart = t  # local t and not account for scr refresh
                yellow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow.started')
                # update status
                yellow.status = STARTED
                yellow.setAutoDraw(True)
            
            # if yellow is active this frame...
            if yellow.status == STARTED:
                # update params
                pass
            
            # if yellow is stopping this frame...
            if yellow.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > yellow.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    yellow.tStop = t  # not accounting for scr refresh
                    yellow.tStopRefresh = tThisFlipGlobal  # on global time
                    yellow.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'yellow.stopped')
                    # update status
                    yellow.status = FINISHED
                    yellow.setAutoDraw(False)
            
            # *blue* updates
            
            # if blue is starting this frame...
            if blue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blue.frameNStart = frameN  # exact frame index
                blue.tStart = t  # local t and not account for scr refresh
                blue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue.started')
                # update status
                blue.status = STARTED
                blue.setAutoDraw(True)
            
            # if blue is active this frame...
            if blue.status == STARTED:
                # update params
                pass
            
            # if blue is stopping this frame...
            if blue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blue.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    blue.tStop = t  # not accounting for scr refresh
                    blue.tStopRefresh = tThisFlipGlobal  # on global time
                    blue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blue.stopped')
                    # update status
                    blue.status = FINISHED
                    blue.setAutoDraw(False)
            
            # *choose_bandit* updates
            waitOnFlip = False
            
            # if choose_bandit is starting this frame...
            if choose_bandit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                choose_bandit.frameNStart = frameN  # exact frame index
                choose_bandit.tStart = t  # local t and not account for scr refresh
                choose_bandit.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choose_bandit, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choose_bandit.started')
                # update status
                choose_bandit.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(choose_bandit.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(choose_bandit.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if choose_bandit is stopping this frame...
            if choose_bandit.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > choose_bandit.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    choose_bandit.tStop = t  # not accounting for scr refresh
                    choose_bandit.tStopRefresh = tThisFlipGlobal  # on global time
                    choose_bandit.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'choose_bandit.stopped')
                    # update status
                    choose_bandit.status = FINISHED
                    choose_bandit.status = FINISHED
            if choose_bandit.status == STARTED and not waitOnFlip:
                theseKeys = choose_bandit.getKeys(keyList=['r', 'f', 'i', 'j'], ignoreKeys=["escape"], waitRelease=False)
                _choose_bandit_allKeys.extend(theseKeys)
                if len(_choose_bandit_allKeys):
                    choose_bandit.keys = _choose_bandit_allKeys[-1].name  # just the last key pressed
                    choose_bandit.rt = _choose_bandit_allKeys[-1].rt
                    choose_bandit.duration = _choose_bandit_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *Fixation* updates
            
            # if Fixation is starting this frame...
            if Fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Fixation.frameNStart = frameN  # exact frame index
                Fixation.tStart = t  # local t and not account for scr refresh
                Fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fixation.started')
                # update status
                Fixation.status = STARTED
                Fixation.setAutoDraw(True)
            
            # if Fixation is active this frame...
            if Fixation.status == STARTED:
                # update params
                pass
            
            # if Fixation is stopping this frame...
            if Fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Fixation.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Fixation.tStop = t  # not accounting for scr refresh
                    Fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    Fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fixation.stopped')
                    # update status
                    Fixation.status = FINISHED
                    Fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                choose.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in choose.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "choose" ---
        for thisComponent in choose.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for choose
        choose.tStop = globalClock.getTime(format='float')
        choose.tStopRefresh = tThisFlipGlobal
        thisExp.addData('choose.stopped', choose.tStop)
        # check responses
        if choose_bandit.keys in ['', [], None]:  # No response was made
            choose_bandit.keys = None
        trials.addData('choose_bandit.keys',choose_bandit.keys)
        if choose_bandit.keys != None:  # we had a response
            trials.addData('choose_bandit.rt', choose_bandit.rt)
            trials.addData('choose_bandit.duration', choose_bandit.duration)
        # Run 'End Routine' code from choose_code
        trialsN = trialsN + 1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if choose.maxDurationReached:
            routineTimer.addTime(-choose.maxDuration)
        elif choose.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "result" ---
        # create an object to store info about Routine result
        result = data.Routine(
            name='result',
            components=[black, red_2, green_2, yellow_2, blue_2, fixation_2, docs, image_2],
        )
        result.status = NOT_STARTED
        continueRoutine = True
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
        # store start times for result
        result.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        result.tStart = globalClock.getTime(format='float')
        result.status = STARTED
        thisExp.addData('result.started', result.tStart)
        result.maxDuration = None
        # keep track of which components have finished
        resultComponents = result.components
        for thisComponent in result.components:
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
        
        # --- Run Routine "result" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        result.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black* updates
            
            # if black is starting this frame...
            if black.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                black.frameNStart = frameN  # exact frame index
                black.tStart = t  # local t and not account for scr refresh
                black.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black.started')
                # update status
                black.status = STARTED
                black.setAutoDraw(True)
            
            # if black is active this frame...
            if black.status == STARTED:
                # update params
                pass
            
            # if black is stopping this frame...
            if black.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black.tStartRefresh + f-frameTolerance:
                    # keep track of stop time/frame for later
                    black.tStop = t  # not accounting for scr refresh
                    black.tStopRefresh = tThisFlipGlobal  # on global time
                    black.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black.stopped')
                    # update status
                    black.status = FINISHED
                    black.setAutoDraw(False)
            
            # *red_2* updates
            
            # if red_2 is starting this frame...
            if red_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                red_2.frameNStart = frameN  # exact frame index
                red_2.tStart = t  # local t and not account for scr refresh
                red_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_2.started')
                # update status
                red_2.status = STARTED
                red_2.setAutoDraw(True)
            
            # if red_2 is active this frame...
            if red_2.status == STARTED:
                # update params
                pass
            
            # if red_2 is stopping this frame...
            if red_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > red_2.tStartRefresh + f-frameTolerance:
                    # keep track of stop time/frame for later
                    red_2.tStop = t  # not accounting for scr refresh
                    red_2.tStopRefresh = tThisFlipGlobal  # on global time
                    red_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_2.stopped')
                    # update status
                    red_2.status = FINISHED
                    red_2.setAutoDraw(False)
            
            # *green_2* updates
            
            # if green_2 is starting this frame...
            if green_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                green_2.frameNStart = frameN  # exact frame index
                green_2.tStart = t  # local t and not account for scr refresh
                green_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_2.started')
                # update status
                green_2.status = STARTED
                green_2.setAutoDraw(True)
            
            # if green_2 is active this frame...
            if green_2.status == STARTED:
                # update params
                pass
            
            # if green_2 is stopping this frame...
            if green_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > green_2.tStartRefresh + f-frameTolerance:
                    # keep track of stop time/frame for later
                    green_2.tStop = t  # not accounting for scr refresh
                    green_2.tStopRefresh = tThisFlipGlobal  # on global time
                    green_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_2.stopped')
                    # update status
                    green_2.status = FINISHED
                    green_2.setAutoDraw(False)
            
            # *yellow_2* updates
            
            # if yellow_2 is starting this frame...
            if yellow_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                yellow_2.frameNStart = frameN  # exact frame index
                yellow_2.tStart = t  # local t and not account for scr refresh
                yellow_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellow_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow_2.started')
                # update status
                yellow_2.status = STARTED
                yellow_2.setAutoDraw(True)
            
            # if yellow_2 is active this frame...
            if yellow_2.status == STARTED:
                # update params
                pass
            
            # if yellow_2 is stopping this frame...
            if yellow_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > yellow_2.tStartRefresh + f-frameTolerance:
                    # keep track of stop time/frame for later
                    yellow_2.tStop = t  # not accounting for scr refresh
                    yellow_2.tStopRefresh = tThisFlipGlobal  # on global time
                    yellow_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'yellow_2.stopped')
                    # update status
                    yellow_2.status = FINISHED
                    yellow_2.setAutoDraw(False)
            
            # *blue_2* updates
            
            # if blue_2 is starting this frame...
            if blue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blue_2.frameNStart = frameN  # exact frame index
                blue_2.tStart = t  # local t and not account for scr refresh
                blue_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_2.started')
                # update status
                blue_2.status = STARTED
                blue_2.setAutoDraw(True)
            
            # if blue_2 is active this frame...
            if blue_2.status == STARTED:
                # update params
                pass
            
            # if blue_2 is stopping this frame...
            if blue_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blue_2.tStartRefresh + f-frameTolerance:
                    # keep track of stop time/frame for later
                    blue_2.tStop = t  # not accounting for scr refresh
                    blue_2.tStopRefresh = tThisFlipGlobal  # on global time
                    blue_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blue_2.stopped')
                    # update status
                    blue_2.status = FINISHED
                    blue_2.setAutoDraw(False)
            
            # *fixation_2* updates
            
            # if fixation_2 is starting this frame...
            if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_2.frameNStart = frameN  # exact frame index
                fixation_2.tStart = t  # local t and not account for scr refresh
                fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2.started')
                # update status
                fixation_2.status = STARTED
                fixation_2.setAutoDraw(True)
            
            # if fixation_2 is active this frame...
            if fixation_2.status == STARTED:
                # update params
                pass
            
            # if fixation_2 is stopping this frame...
            if fixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_2.tStartRefresh + f-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_2.tStop = t  # not accounting for scr refresh
                    fixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_2.stopped')
                    # update status
                    fixation_2.status = FINISHED
                    fixation_2.setAutoDraw(False)
            
            # *docs* updates
            
            # if docs is starting this frame...
            if docs.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                docs.frameNStart = frameN  # exact frame index
                docs.tStart = t  # local t and not account for scr refresh
                docs.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(docs, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'docs.started')
                # update status
                docs.status = STARTED
                docs.setAutoDraw(True)
            
            # if docs is active this frame...
            if docs.status == STARTED:
                # update params
                pass
            
            # if docs is stopping this frame...
            if docs.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > docs.tStartRefresh + z-frameTolerance:
                    # keep track of stop time/frame for later
                    docs.tStop = t  # not accounting for scr refresh
                    docs.tStopRefresh = tThisFlipGlobal  # on global time
                    docs.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'docs.stopped')
                    # update status
                    docs.status = FINISHED
                    docs.setAutoDraw(False)
            
            # *image_2* updates
            
            # if image_2 is starting this frame...
            if image_2.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                image_2.frameNStart = frameN  # exact frame index
                image_2.tStart = t  # local t and not account for scr refresh
                image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.started')
                # update status
                image_2.status = STARTED
                image_2.setAutoDraw(True)
            
            # if image_2 is active this frame...
            if image_2.status == STARTED:
                # update params
                pass
            
            # if image_2 is stopping this frame...
            if image_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_2.tStartRefresh + z-frameTolerance:
                    # keep track of stop time/frame for later
                    image_2.tStop = t  # not accounting for scr refresh
                    image_2.tStopRefresh = tThisFlipGlobal  # on global time
                    image_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.stopped')
                    # update status
                    image_2.status = FINISHED
                    image_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                result.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in result.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "result" ---
        for thisComponent in result.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for result
        result.tStop = globalClock.getTime(format='float')
        result.tStopRefresh = tThisFlipGlobal
        thisExp.addData('result.stopped', result.tStop)
        # the Routine "result" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "comparison_result" ---
        # create an object to store info about Routine comparison_result
        comparison_result = data.Routine(
            name='comparison_result',
            components=[black_com, reward_sub, palyerb, reward__sub, compare, red_compare, green_com, yellow_com, blue_com],
        )
        comparison_result.status = NOT_STARTED
        continueRoutine = True
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
        # store start times for comparison_result
        comparison_result.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        comparison_result.tStart = globalClock.getTime(format='float')
        comparison_result.status = STARTED
        thisExp.addData('comparison_result.started', comparison_result.tStart)
        comparison_result.maxDuration = None
        # keep track of which components have finished
        comparison_resultComponents = comparison_result.components
        for thisComponent in comparison_result.components:
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
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        comparison_result.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_com* updates
            
            # if black_com is starting this frame...
            if black_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_com.frameNStart = frameN  # exact frame index
                black_com.tStart = t  # local t and not account for scr refresh
                black_com.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_com, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_com.started')
                # update status
                black_com.status = STARTED
                black_com.setAutoDraw(True)
            
            # if black_com is active this frame...
            if black_com.status == STARTED:
                # update params
                pass
            
            # if black_com is stopping this frame...
            if black_com.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_com.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    black_com.tStop = t  # not accounting for scr refresh
                    black_com.tStopRefresh = tThisFlipGlobal  # on global time
                    black_com.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_com.stopped')
                    # update status
                    black_com.status = FINISHED
                    black_com.setAutoDraw(False)
            
            # *reward_sub* updates
            
            # if reward_sub is starting this frame...
            if reward_sub.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward_sub.frameNStart = frameN  # exact frame index
                reward_sub.tStart = t  # local t and not account for scr refresh
                reward_sub.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward_sub, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_sub.started')
                # update status
                reward_sub.status = STARTED
                reward_sub.setAutoDraw(True)
            
            # if reward_sub is active this frame...
            if reward_sub.status == STARTED:
                # update params
                pass
            
            # if reward_sub is stopping this frame...
            if reward_sub.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reward_sub.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    reward_sub.tStop = t  # not accounting for scr refresh
                    reward_sub.tStopRefresh = tThisFlipGlobal  # on global time
                    reward_sub.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reward_sub.stopped')
                    # update status
                    reward_sub.status = FINISHED
                    reward_sub.setAutoDraw(False)
            
            # *palyerb* updates
            
            # if palyerb is starting this frame...
            if palyerb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                palyerb.frameNStart = frameN  # exact frame index
                palyerb.tStart = t  # local t and not account for scr refresh
                palyerb.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(palyerb, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'palyerb.started')
                # update status
                palyerb.status = STARTED
                palyerb.setAutoDraw(True)
            
            # if palyerb is active this frame...
            if palyerb.status == STARTED:
                # update params
                pass
            
            # if palyerb is stopping this frame...
            if palyerb.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > palyerb.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    palyerb.tStop = t  # not accounting for scr refresh
                    palyerb.tStopRefresh = tThisFlipGlobal  # on global time
                    palyerb.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'palyerb.stopped')
                    # update status
                    palyerb.status = FINISHED
                    palyerb.setAutoDraw(False)
            
            # *reward__sub* updates
            
            # if reward__sub is starting this frame...
            if reward__sub.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward__sub.frameNStart = frameN  # exact frame index
                reward__sub.tStart = t  # local t and not account for scr refresh
                reward__sub.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward__sub, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward__sub.started')
                # update status
                reward__sub.status = STARTED
                reward__sub.setAutoDraw(True)
            
            # if reward__sub is active this frame...
            if reward__sub.status == STARTED:
                # update params
                pass
            
            # if reward__sub is stopping this frame...
            if reward__sub.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reward__sub.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    reward__sub.tStop = t  # not accounting for scr refresh
                    reward__sub.tStopRefresh = tThisFlipGlobal  # on global time
                    reward__sub.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reward__sub.stopped')
                    # update status
                    reward__sub.status = FINISHED
                    reward__sub.setAutoDraw(False)
            
            # *compare* updates
            
            # if compare is starting this frame...
            if compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                compare.frameNStart = frameN  # exact frame index
                compare.tStart = t  # local t and not account for scr refresh
                compare.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(compare, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'compare.started')
                # update status
                compare.status = STARTED
                compare.setAutoDraw(True)
            
            # if compare is active this frame...
            if compare.status == STARTED:
                # update params
                pass
            
            # if compare is stopping this frame...
            if compare.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > compare.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    compare.tStop = t  # not accounting for scr refresh
                    compare.tStopRefresh = tThisFlipGlobal  # on global time
                    compare.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'compare.stopped')
                    # update status
                    compare.status = FINISHED
                    compare.setAutoDraw(False)
            
            # *red_compare* updates
            
            # if red_compare is starting this frame...
            if red_compare.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                red_compare.frameNStart = frameN  # exact frame index
                red_compare.tStart = t  # local t and not account for scr refresh
                red_compare.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(red_compare, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'red_compare.started')
                # update status
                red_compare.status = STARTED
                red_compare.setAutoDraw(True)
            
            # if red_compare is active this frame...
            if red_compare.status == STARTED:
                # update params
                pass
            
            # if red_compare is stopping this frame...
            if red_compare.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > red_compare.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    red_compare.tStop = t  # not accounting for scr refresh
                    red_compare.tStopRefresh = tThisFlipGlobal  # on global time
                    red_compare.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'red_compare.stopped')
                    # update status
                    red_compare.status = FINISHED
                    red_compare.setAutoDraw(False)
            
            # *green_com* updates
            
            # if green_com is starting this frame...
            if green_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                green_com.frameNStart = frameN  # exact frame index
                green_com.tStart = t  # local t and not account for scr refresh
                green_com.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(green_com, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'green_com.started')
                # update status
                green_com.status = STARTED
                green_com.setAutoDraw(True)
            
            # if green_com is active this frame...
            if green_com.status == STARTED:
                # update params
                pass
            
            # if green_com is stopping this frame...
            if green_com.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > green_com.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    green_com.tStop = t  # not accounting for scr refresh
                    green_com.tStopRefresh = tThisFlipGlobal  # on global time
                    green_com.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'green_com.stopped')
                    # update status
                    green_com.status = FINISHED
                    green_com.setAutoDraw(False)
            
            # *yellow_com* updates
            
            # if yellow_com is starting this frame...
            if yellow_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                yellow_com.frameNStart = frameN  # exact frame index
                yellow_com.tStart = t  # local t and not account for scr refresh
                yellow_com.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellow_com, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellow_com.started')
                # update status
                yellow_com.status = STARTED
                yellow_com.setAutoDraw(True)
            
            # if yellow_com is active this frame...
            if yellow_com.status == STARTED:
                # update params
                pass
            
            # if yellow_com is stopping this frame...
            if yellow_com.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > yellow_com.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    yellow_com.tStop = t  # not accounting for scr refresh
                    yellow_com.tStopRefresh = tThisFlipGlobal  # on global time
                    yellow_com.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'yellow_com.stopped')
                    # update status
                    yellow_com.status = FINISHED
                    yellow_com.setAutoDraw(False)
            
            # *blue_com* updates
            
            # if blue_com is starting this frame...
            if blue_com.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blue_com.frameNStart = frameN  # exact frame index
                blue_com.tStart = t  # local t and not account for scr refresh
                blue_com.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blue_com, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blue_com.started')
                # update status
                blue_com.status = STARTED
                blue_com.setAutoDraw(True)
            
            # if blue_com is active this frame...
            if blue_com.status == STARTED:
                # update params
                pass
            
            # if blue_com is stopping this frame...
            if blue_com.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blue_com.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    blue_com.tStop = t  # not accounting for scr refresh
                    blue_com.tStopRefresh = tThisFlipGlobal  # on global time
                    blue_com.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blue_com.stopped')
                    # update status
                    blue_com.status = FINISHED
                    blue_com.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                comparison_result.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in comparison_result.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "comparison_result" ---
        for thisComponent in comparison_result.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for comparison_result
        comparison_result.tStop = globalClock.getTime(format='float')
        comparison_result.tStopRefresh = tThisFlipGlobal
        thisExp.addData('comparison_result.stopped', comparison_result.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if comparison_result.maxDurationReached:
            routineTimer.addTime(-comparison_result.maxDuration)
        elif comparison_result.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "slide" ---
        # create an object to store info about Routine slide
        slide = data.Routine(
            name='slide',
            components=[image_3, slider, text, image_4, image_5],
        )
        slide.status = NOT_STARTED
        continueRoutine = True
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
        # store start times for slide
        slide.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        slide.tStart = globalClock.getTime(format='float')
        slide.status = STARTED
        thisExp.addData('slide.started', slide.tStart)
        slide.maxDuration = None
        # keep track of which components have finished
        slideComponents = slide.components
        for thisComponent in slide.components:
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
        
        # --- Run Routine "slide" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        slide.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_3* updates
            
            # if image_3 is starting this frame...
            if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_3.frameNStart = frameN  # exact frame index
                image_3.tStart = t  # local t and not account for scr refresh
                image_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_3.started')
                # update status
                image_3.status = STARTED
                image_3.setAutoDraw(True)
            
            # if image_3 is active this frame...
            if image_3.status == STARTED:
                # update params
                pass
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # Check slider for response to end Routine
            if slider.getRating() is not None and slider.status == STARTED:
                continueRoutine = False
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *image_4* updates
            
            # if image_4 is starting this frame...
            if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_4.frameNStart = frameN  # exact frame index
                image_4.tStart = t  # local t and not account for scr refresh
                image_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_4.started')
                # update status
                image_4.status = STARTED
                image_4.setAutoDraw(True)
            
            # if image_4 is active this frame...
            if image_4.status == STARTED:
                # update params
                pass
            
            # *image_5* updates
            
            # if image_5 is starting this frame...
            if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_5.frameNStart = frameN  # exact frame index
                image_5.tStart = t  # local t and not account for scr refresh
                image_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_5.started')
                # update status
                image_5.status = STARTED
                image_5.setAutoDraw(True)
            
            # if image_5 is active this frame...
            if image_5.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                slide.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in slide.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "slide" ---
        for thisComponent in slide.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for slide
        slide.tStop = globalClock.getTime(format='float')
        slide.tStopRefresh = tThisFlipGlobal
        thisExp.addData('slide.stopped', slide.tStop)
        # Run 'End Routine' code from randomwalk
        thisExp.addData('rewardyellow', str(rewardyellowlist[0]))
        thisExp.addData('rewardred', str(rewardredlist[0]))
        thisExp.addData('rewardblue', str(rewardbluelist[0]))
        thisExp.addData('rewardgreen', str(rewardgreenlist[0]))
        thisExp.addData('subchoose', str(alist[0]))
        thisExp.addData('playerb', str(jlist[0]))
        trials.addData('slider.response', slider.getRating())
        trials.addData('slider.rt', slider.getRT())
        # the Routine "slide" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wrongsign" ---
        # create an object to store info about Routine wrongsign
        wrongsign = data.Routine(
            name='wrongsign',
            components=[bigx],
        )
        wrongsign.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
            i=0
            continueRoutine = False
        else:
            i=4.5
            continueRoutine = True
        bigx.setText('×')
        # store start times for wrongsign
        wrongsign.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        wrongsign.tStart = globalClock.getTime(format='float')
        wrongsign.status = STARTED
        thisExp.addData('wrongsign.started', wrongsign.tStart)
        wrongsign.maxDuration = None
        # keep track of which components have finished
        wrongsignComponents = wrongsign.components
        for thisComponent in wrongsign.components:
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
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        wrongsign.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bigx* updates
            
            # if bigx is starting this frame...
            if bigx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bigx.frameNStart = frameN  # exact frame index
                bigx.tStart = t  # local t and not account for scr refresh
                bigx.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bigx, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bigx.started')
                # update status
                bigx.status = STARTED
                bigx.setAutoDraw(True)
            
            # if bigx is active this frame...
            if bigx.status == STARTED:
                # update params
                pass
            
            # if bigx is stopping this frame...
            if bigx.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bigx.tStartRefresh + i-frameTolerance:
                    # keep track of stop time/frame for later
                    bigx.tStop = t  # not accounting for scr refresh
                    bigx.tStopRefresh = tThisFlipGlobal  # on global time
                    bigx.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bigx.stopped')
                    # update status
                    bigx.status = FINISHED
                    bigx.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                wrongsign.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wrongsign.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wrongsign" ---
        for thisComponent in wrongsign.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for wrongsign
        wrongsign.tStop = globalClock.getTime(format='float')
        wrongsign.tStopRefresh = tThisFlipGlobal
        thisExp.addData('wrongsign.stopped', wrongsign.tStop)
        # the Routine "wrongsign" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "intertrial" ---
        # create an object to store info about Routine intertrial
        intertrial = data.Routine(
            name='intertrial',
            components=[intertrialfix],
        )
        intertrial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        import numpy as np
        intertrialtime = np.random.normal(2, 1)
        while (intertrialtime < 1.5) | (intertrialtime > 2):
            intertrialtime = np.random.normal(2,1)
        # store start times for intertrial
        intertrial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        intertrial.tStart = globalClock.getTime(format='float')
        intertrial.status = STARTED
        thisExp.addData('intertrial.started', intertrial.tStart)
        intertrial.maxDuration = None
        # keep track of which components have finished
        intertrialComponents = intertrial.components
        for thisComponent in intertrial.components:
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
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        intertrial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *intertrialfix* updates
            
            # if intertrialfix is starting this frame...
            if intertrialfix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                intertrialfix.frameNStart = frameN  # exact frame index
                intertrialfix.tStart = t  # local t and not account for scr refresh
                intertrialfix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(intertrialfix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'intertrialfix.started')
                # update status
                intertrialfix.status = STARTED
                intertrialfix.setAutoDraw(True)
            
            # if intertrialfix is active this frame...
            if intertrialfix.status == STARTED:
                # update params
                pass
            
            # if intertrialfix is stopping this frame...
            if intertrialfix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > intertrialfix.tStartRefresh + intertrialtime-frameTolerance:
                    # keep track of stop time/frame for later
                    intertrialfix.tStop = t  # not accounting for scr refresh
                    intertrialfix.tStopRefresh = tThisFlipGlobal  # on global time
                    intertrialfix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'intertrialfix.stopped')
                    # update status
                    intertrialfix.status = FINISHED
                    intertrialfix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                intertrial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in intertrial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "intertrial" ---
        for thisComponent in intertrial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for intertrial
        intertrial.tStop = globalClock.getTime(format='float')
        intertrial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('intertrial.stopped', intertrial.tStop)
        # the Routine "intertrial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 150 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[image, key_resp],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
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
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 30.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image* updates
        
        # if image is starting this frame...
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.started')
            # update status
            image.status = STARTED
            image.setAutoDraw(True)
        
        # if image is active this frame...
        if image.status == STARTED:
            # update params
            pass
        
        # if image is stopping this frame...
        if image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.tStopRefresh = tThisFlipGlobal  # on global time
                image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.stopped')
                # update status
                image.status = FINISHED
                image.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.stopped')
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-30.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
