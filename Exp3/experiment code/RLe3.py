#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.5),
    on Tue Apr 18 23:42:17 2023
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
import random
tempArray = ["arm1", "arm2", "arm3", "arm4"]
random.shuffle(tempArray)


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.5'
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
    originPath='/Users/wututu/Library/CloudStorage/OneDrive-UniversityofMacau/RL_E3/RLe3.py',
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
    size=[1512, 982], fullscr=True, screen=0, 
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
    image='pic/intro1.png', mask=None, anchor='center',
    ori=0, pos=(0, 0), size=(1.6,1),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
instru_resp = keyboard.Keyboard()

# --- Initialize components for Routine "instr2" ---
instru_img2 = visual.ImageStim(
    win=win,
    name='instru_img2', 
    image='pic/intro22.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1.67,1),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
instru_resp2 = keyboard.Keyboard()

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

# --- Initialize components for Routine "result" ---
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
    ori=0.0, pos=(0, 0.3), size=(0.365,0.1),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
reward__sub = visual.TextStim(win=win, name='reward__sub',
    text='',
    font='Open Sans',
    pos=(0.0647, 0.305), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
red_compare = visual.ImageStim(
    win=win,
    name='red_compare', 
    image='pic/red.png', mask=None, anchor='center',
    ori=0, pos=(-0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)
green_com = visual.ImageStim(
    win=win,
    name='green_com', 
    image='pic/green.png', mask=None, anchor='center',
    ori=0, pos=(0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-5.0)
yellow_com = visual.ImageStim(
    win=win,
    name='yellow_com', 
    image='pic/yellow.png', mask=None, anchor='center',
    ori=0, pos=(-0.6, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-6.0)
blue_com = visual.ImageStim(
    win=win,
    name='blue_com', 
    image='pic/blue.png', mask=None, anchor='center',
    ori=0, pos=(0.2, 0), size=[0.3],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-7.0)

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
    trialList=data.importConditions('reward_distribution_e3/reward_distribution_1.csv'),
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
    import random
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        z = np.random.uniform(0.8,2)#生成随机小数作为result的duration
        f = z+0.3#图片呈现时间 
    else:
        f=0
        continueRoutine = False
        
    redpoints = locals()[tempArray[0]]
    greenpoints = locals()[tempArray[1]]
    bluepoints = locals()[tempArray[2]]
    yellowpoints = locals()[tempArray[3]]
    if choose_bandit.keys == 'r':#如果被试选择某一个bandit，就显示对应的值
        black_x_pos = -0.6
        black_y_pos = 0.
        docs_x_pos = -0.6
        docs_y_pos = 0
        points = yellowpoints
        ss = 300
    elif choose_bandit.keys == 'f':
        black_x_pos = -0.2
        black_y_pos = 0
        docs_x_pos = -0.2
        docs_y_pos = 0
        ss = 300
        points = redpoints
    elif choose_bandit.keys == 'i':
        black_x_pos = 0.2
        black_y_pos = 0
        docs_x_pos = 0.2
        docs_y_pos = 0
        ss = 300
        points = bluepoints
    elif choose_bandit.keys == 'j':
        black_x_pos = 0.6
        black_y_pos = 0
        docs_x_pos = 0.6
        docs_y_pos = 0
        ss = 300
        points = greenpoints
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
    rewardyellowlist.append(yellowpoints)
    rewardredlist.append(redpoints)
    rewardbluelist.append(bluepoints)
    rewardgreenlist.append(greenpoints)
    
    black.setPos((black_x_pos, black_y_pos))
    black.setImage('pic/black.png')
    red_2.setImage('pic/red.png')
    green_2.setImage('pic/green.png')
    yellow_2.setImage('pic/yellow.png')
    blue_2.setImage('pic/blue.png')
    fixation_2.setText('+')
    docs.setPos((docs_x_pos,docs_y_pos))
    docs.setText('● ● ●')
    # keep track of which components have finished
    choose_resultComponents = [black, red_2, green_2, yellow_2, blue_2, fixation_2, docs]
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
    
    # --- Prepare to start Routine "result" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from comparisonresult_code
    if (choose_bandit.keys == 'r')|(choose_bandit.keys == 'f') | (choose_bandit.keys == 'i') | (choose_bandit.keys == 'j'):
        continueRoutine = True
        a = points
    else:
        continueRoutine = False
        a = 0
        
    alist=[]
    alist.append(a)
    black_com.setPos((black_x_pos, black_y_pos))
    reward__sub.setText(a)
    # keep track of which components have finished
    resultComponents = [black_com, reward_sub, reward__sub, red_compare, green_com, yellow_com, blue_com]
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
    frameN = -1
    
    # --- Run Routine "result" ---
    while continueRoutine:
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
            if tThisFlipGlobal > black_com.tStartRefresh + f-frameTolerance:
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
            if tThisFlipGlobal > reward_sub.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                reward_sub.tStop = t  # not accounting for scr refresh
                reward_sub.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_sub.stopped')
                reward_sub.setAutoDraw(False)
        
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
            if tThisFlipGlobal > reward__sub.tStartRefresh + f-frameTolerance:
                # keep track of stop time/frame for later
                reward__sub.tStop = t  # not accounting for scr refresh
                reward__sub.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward__sub.stopped')
                reward__sub.setAutoDraw(False)
        
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
            if tThisFlipGlobal > red_compare.tStartRefresh + f-frameTolerance:
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
            if tThisFlipGlobal > green_com.tStartRefresh + f-frameTolerance:
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
            if tThisFlipGlobal > yellow_com.tStartRefresh + f-frameTolerance:
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
            if tThisFlipGlobal > blue_com.tStartRefresh + f-frameTolerance:
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
        for thisComponent in resultComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "result" ---
    for thisComponent in resultComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Run 'End Routine' code from comparisonresult_code
    thisExp.addData('rewardred',redpoints)
    thisExp.addData('rewardyellow',yellowpoints)
    thisExp.addData('rewardblue',bluepoints)
    thisExp.addData('rewardgreen',greenpoints)
    thisExp.addData('subchoose',a)
    # the Routine "result" was not non-slip safe, so reset the non-slip timer
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
