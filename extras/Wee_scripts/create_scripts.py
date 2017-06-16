# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:24:32 2015

@author: ajaver
"""


class WeeScript:
    def __init__(self, name=''):
        self.xml_cmd = '''<?xml version="1.0"?>
        <Experiment xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" Name="%s" Description="">
          <WormRigEvents>
        ''' % name
    
    def gas_event(self, N2=0, O2=0, CO2=0):
        self.xml_cmd += '    <WormRigEvent xsi:type="GasEvent">\n' + \
        '      <Nitrogen Name="N" Description="Nitrogen" Percentage="%i" />\n' % N2 + \
        '      <Oxygen Name="O₂" Description="Oxygen" Percentage="%i" />\n' % O2 + \
        '      <CarbonDioxide Name="CO₂" Description="Carbon Dioxide" Percentage="%i" />\n' % CO2+ \
        '    </WormRigEvent>\n'

    def move_event(self, position):
        self.xml_cmd += ('    <WormRigEvent xsi:type="MovePositionEvent">\n' \
                        '      <RigPosition>%s</RigPosition>\n' + \
                        '    </WormRigEvent>\n') % position
    
    def vision_event(self, camera_state):
        assert camera_state != 'Start' or camera_state != 'Stop'
        self.xml_cmd += '    <WormRigEvent xsi:type="VisionEvent" Operation="%s" />\n' % camera_state
    
    def wait_event(self, minutes):
        assert isinstance(minutes, int) 
        self.xml_cmd += '    <WormRigEvent xsi:type="WaitEvent" Duration="%i" />\n' % minutes
    
    def close(self):
        self.xml_cmd += '  </WormRigEvents>\n</Experiment>\n'

    
def step():

    delT = 3
    Nloops = 2
    dd = WeeScript('GAS_RAMP')

    dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
    dd.wait_event(delT)

    for kk in range(Nloops):    
        dd.gas_event(N2 = 91, O2 = 10)
        dd.wait_event(delT)
        dd.gas_event(N2 = 91, O2 = 35)
        dd.wait_event(delT)

    dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
    dd.wait_event(delT)

    #close the valves    
    dd.gas_event(N2 = 0, O2 = 0)
    dd.close()
    with open('STEPS.xml', 'w') as f:
        f.write(dd.xml_cmd)



def ramp():
    
    dd = WeeScript('GAS_RAMP')
#    dd.gas_event(N2 = 0, O2 = 0)
#    dd.wait_event(5)
#    
#    dd.gas_event(N2 = 91, O2 = 0)
#    dd.wait_event(10)
#        
#    for kk in range(5):    
#        dd.gas_event(N2 = 91, O2 = 10)
#        dd.wait_event(7)
#        dd.gas_event(N2 = 91, O2 = 35)
#        dd.wait_event(7)
#    
#    dd.gas_event(N2 = 0, O2 = 0)
#    dd.wait_event(3)
    
    ramp_bot = 0
    ramp_top = 40    
    ramp_del = 2
    ramp_wait = 1
    
    dd.gas_event(N2 = 91, O2 = 0)
    dd.wait_event(3)
    
    ramp = list(range(ramp_bot, ramp_top+ramp_del, ramp_del)) + list(range(ramp_top-ramp_del, ramp_bot, -ramp_del)) 
    ramp = 3*ramp + [ramp_bot]
    for kk in ramp:
        dd.gas_event(N2 = 91, O2 = kk)
        dd.wait_event(ramp_wait)
    
    dd.gas_event(N2 = 0, O2 = 0)
    dd.wait_event(5)
    dd.close()
        
    with open('GAS_RAMP.xml', 'w') as f:
        f.write(dd.xml_cmd)
    
    

def exp_type1():
    dd = WeeScript('RecordDiffPos')
    
    for kk in range(23):
        for plate_N in ['Plate4', 'Plate3']:
            dd.move_event(plate_N)
            dd.vision_event('Start')
            dd.wait_event(5)
            
            dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
            dd.wait_event(15)
            
            dd.gas_event(N2 = 91, O2 = 40) #high oxygen
            dd.wait_event(15)
            
            dd.gas_event(N2 = 91, O2 = 0) #low oxygen
            dd.wait_event(15)
            
            dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
            dd.wait_event(10)
            
            dd.gas_event(N2 = 0, O2 = 0) #aprox atmospheric oxygen
            dd.wait_event(5)
            
            
            dd.vision_event('Stop')
            
            
            dd.wait_event(1)    

    dd.close()
    with open('REC_O2.xml', 'w') as f:
        f.write(dd.xml_cmd)


def exp_type2():
    dd = WeeScript('RecordDiffPos')
    
    for kk in range(23):
        dd.move_event('Plate3')
        dd.vision_event('Start')
        
        dd.gas_event(N2 = 0, O2 = 0) #close all valves
        dd.wait_event(30)
        
        dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
        dd.wait_event(15)
        
        dd.gas_event(N2 = 91, O2 = 40) #high oxygen
        dd.wait_event(15)
        
        dd.gas_event(N2 = 91, O2 = 0) #low oxygen
        dd.wait_event(15)
        
        dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
        dd.wait_event(5)
        
        dd.gas_event(N2 = 91, O2 = 24, CO2 = 3) #high CO2
        dd.wait_event(15)

        dd.gas_event(N2 = 91, O2 = 24) #aprox atmospheric oxygen
        dd.wait_event(10)
        
        dd.gas_event(N2 = 0, O2 = 0) #close all valves
        dd.wait_event(5)
        
        dd.vision_event('Stop')
        
        #dd.move_event('Plate3')
        #dd.vision_event('Start')
        #dd.wait_event(15)
        #dd.vision_event('Stop')


    dd.close()
    with open('REC_O2_type2.xml', 'w') as f:
        f.write(dd.xml_cmd)

def video():
    dd = WeeScript('RecordDiffPos')
    for kk in range(24):    
        for n_pos in ['Plate1', 'Plate8']:
            dd.move_event(n_pos)        
            dd.vision_event('Start')
            dd.wait_event(60)
            dd.vision_event('Stop')
        
    dd.close()
    with open('RecordDiffPos.xml', 'w') as f:
        f.write(dd.xml_cmd)

def timelaps():
    record_time = 30
    wait_time = 6*60
    plates2record = ['Plate7', 'Plate6', 'Plate5', 'Plate4', 'Plate3', 'Plate2']    
    
    dd = WeeScript('Timelaps')
    for kk in range(24): 
        for n_pos in plates2record:
                dd.move_event(n_pos)        
                dd.vision_event('Start')
                dd.wait_event(record_time)
                dd.vision_event('Stop')
        
        dd.move_event('Plate1')
        dd.wait_event(wait_time - record_time*len(plates2record)) 
        
        
    dd.close()
    with open('Timelaps_6H.xml', 'w') as f:
        f.write(dd.xml_cmd)

if __name__ == '__main__':
    script_name = 'SerenasRamp'
    
    script = WeeScript(script_name)
    
    max_flow = 90
    def get_flows(O2_percent):
        O2_flow = round(O2_percent/100*max_flow)
        N2_flow = max_flow - O2_flow
        return O2_flow, N2_flow
    
    def _record(time):
        script.vision_event('Start')
        script.wait_event(time)
        script.vision_event('Stop')
    
    #let it equilibrate to approx atmospheric concentration
    O2_flow, N2_flow = get_flows(21)
    script.gas_event(N2 = N2_flow, O2 = O2_flow) #aprox atmospheric oxygen
    script.wait_event(5)
    _record(5)
    
    for O2_percent in [7, 10, 13, 16, 19, 21, 25, 30, 25, 60, 75, 90, 100]:
        O2_flow, N2_flow = get_flows(O2_percent)
        script.gas_event(N2 = N2_flow, O2 = O2_flow) #aprox atmospheric oxygen
        script.wait_event(2)
        _record(5)

    O2_flow, N2_flow = get_flows(21)
    script.gas_event(N2 = N2_flow, O2 = O2_flow) #aprox atmospheric oxygen
    script.wait_event(5)
    _record(5)
    
    #close valves
    script.gas_event(N2 = 0, O2 = 0) #aprox atmospheric oxygen
    
    script.close()
    with open(script_name + '.xml', 'w') as f:
        f.write(script.xml_cmd)

#    step()
#    print('done')

