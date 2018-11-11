import time
from ctypes import windll
from psychopy import parallel
import psychopy labjack library

from psychopy.hardware.labjacks import U3

# Initialize labjack
lj = U3()

# Get calibration info and turn FIO 0 to off (usually starts as ON)
cal_data = lj.getCalibrationData()
if lj.getFIOState(0) == 1:
    lj.setFIOState(0,0) #Make sure we start with the trigger off

# At onset of event toggle FIO 0 to ON
lj.setFIOState(0,1)

# At offset of event toggle FIO 0 to OFF
lj.setFIOState(0,0)



'''
port = parallel.ParallelPort()
print port
port.setData(0)
print port.readData()
port.readPin(2)
port.setPin(2, 10)
port.setData(45)  # sets all pins low
print port.readData()



sendtriggers = True
#set parallel port params
if sendtriggers == True:
    IOport = parallel.ParallelPort(address = '0xD010') #change the address to whatever it is for the port on the comp in use
    holdvalue = 0 #reset the port to this after every trigger to avoid repeats, and reset all pins
    triggercode = 1
    IOport.setData(holdvalue)   # always send a zero value to reset all pins after a trigger to prevent repeated sending
    IOport.setData(triggercode) # this is the code to send a trigger to the parallel port (i.e. to amplifier)
    print IOport.readData()
    IOport.setData(holdvalue)   # always send a zero value to reset all pins after a trigger to prevent repeated sending
    print IOport.readData()

P = windll.inpout32
def sendCode(code):
	P.Out32(0x378, code) # send the event code (could be 1-255)
	time.sleep(0.006) # wait for 6 ms for BioSemi to receive the code
	P.Out32(0x378, 0) # send a code to clear the register
	time.sleep(0.01) # wait for 10 ms
sendCode(1)
'''