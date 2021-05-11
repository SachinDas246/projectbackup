#--------------------- Import Librariesc -------------------------------------------------|
from controller import Robot,Keyboard,Supervisor
import numpy as np
import random
import math
from gym import Env
from gym.spaces import Discrete,Box
#import tensorflow as tf


#---------------------- Initilise Robot --------------------------------------------------|
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

#---------------------- Initilize Devices ------------------------------------------------|


m0 = robot.getDevice('m0')
m1 = robot.getDevice('m1')
m2 = robot.getDevice('m2')
m3 = robot.getDevice('m3')
m4 = robot.getDevice('m4')

p0 = robot.getDevice('p0')
p1 = robot.getDevice('p1')
p2 = robot.getDevice('p2')
p3 = robot.getDevice('p3')
p4 = robot.getDevice('p4')

c1 = robot.getDevice('c1')

gps = robot.getDevice('gps')

#--------------------- Set initial Values -----------------------------------------------|

p0.enable(timestep)
p1.enable(timestep)
p2.enable(timestep)
p3.enable(timestep)
p4.enable(timestep)
c1.enablePresence(timestep)
gps.enable(timestep)

m0.setVelocity(1)
m1.setVelocity(1)
m2.setVelocity(1)
m3.setVelocity(1)
m4.setVelocity(1)



#-------------------- Pre Functions -----------------------------------------------------|
def get_point_distance(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2))


def get_random(min,max,emin,emax):
    v = np.random.uniform(min,max)    
    while v < emax and v>emin :        
        v = np.random.uniform(min,max)
    return v        

def IsMotionComplete(mi0,mi1,mi2,mi3,mi4,fact): #check if motion has compleated
    if mi0*(1-fact) <= p0.getValue() <= mi0*(fact+1):
        if mi1*(1-fact) <= p1.getValue() <= mi1*(fact+1):
            if mi2*(1-fact) <= p2.getValue() <= mi2*(fact+1):
                if mi3*(1-fact) <= p3.getValue() <= mi3*(fact+1):
                    if mi4*(1-fact) <= p4.getValue() <= mi4*(fact+1):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False        
        else:
            return False
    else:
        return False

def PosInit(obj_off):#used for setting initial random values to motor
    # set position of load at random
    min_pickup_drop_distance = 0.12
    lpx = get_random(-0.33,0.33,-0.16,0.16)
    lpz = get_random(-0.33,0.33,-0.16,0.16) 
    robot.getFromDef("load").getField("translation").setSFVec3f([lpx,0.0299986,lpz])   
    dpx = get_random(-0.33,0.33,-0.16,0.16)
    dpz = get_random(-0.33,0.33,-0.16,0.16)
    while get_point_distance([lpx,0.0299986,lpz],[dpx,0.0299986,dpz]) < min_pickup_drop_distance:           
        dpx = get_random(-0.33,0.33,-0.16,0.16)
        dpz = get_random(-0.33,0.33,-0.16,0.16)
            
    
    
    obj_hgh_offset = obj_off #distance from centre of object to side faces
    isoutside = False
    while not isoutside:
        mv0 = np.random.uniform(0,6.2831853072)
        mv1 = np.random.uniform(0,2.0943951024)
        mv2 = np.random.uniform(0,2.0943951024)
        mv3 = np.random.uniform(0,2.0943951024)
        mv4 = np.random.uniform(0,2.0943951024)  
        
        robot.getFromDef("jp0").getField("position").setSFFloat(mv0)
        robot.getFromDef("jp1").getField("position").setSFFloat(mv1)
        robot.getFromDef("jp2").getField("position").setSFFloat(mv2)
        robot.getFromDef("jp3").getField("position").setSFFloat(mv3)
        robot.getFromDef("jp4").getField("position").setSFFloat(mv4)
        
        m0.setPosition(mv0)    
        m1.setPosition(mv1)
        m2.setPosition(mv2)
        m3.setPosition(mv3)
        m4.setPosition(mv4)
        
        
        
        #check if initial position is in restricted area        
        
        robot.step(timestep)
        
        
        tpos = gps.getValues()
        if tpos[1] < 0.01+obj_hgh_offset:
            isoutside = False
            continue
        if (tpos[0] < 0.12 and tpos[0] >-0.12) and (tpos[2] < 0.12 and tpos[2] >-0.12) :
            isoutside = False
            continue
        isoutside = True
        
        break
    #dont wory about quick change in position during initial time steps its because of compensating for restricted region  
    
       
    c = 0# random.randint(0, 1)
    
    # check if connection is possible
    islockable = c1.getPresence()
    
    if c == 1:
        c1.lock()
    else:
        c1.unlock()   
    
    pos = gps.getValues()
    
    
            
    return [mv0,mv1,mv2,mv3,mv4,pos[0],pos[1],pos[2],c,islockable,lpx,0.0299986+obj_hgh_offset,lpz,dpx,0.0299986+obj_hgh_offset,dpz]



def updateRobot(instate):
    m0.setPosition(instate[0])
    m1.setPosition(instate[1])
    m2.setPosition(instate[2])
    m3.setPosition(instate[3])
    m4.setPosition(instate[4])
    if instate[0] == 0 and c1.isLocked():
        c1.unlock()
    elif instate[0] == 1 and not c1.isLocked():
        c1.lock()

    


print(get_point_distance([0,0,0],[1,1,1]))


#----------------- Environment class ----------------------------------------------------|
class tars(Env):
    def __init__(self):
        self.doing = 0  #       \
        self.pos_accuracy = 0.001#\ change in these non state variables will require retraining
        self.obj_off = 0.03 # height offset for connector
        self.step_awaposition#/
        # Action we can take increse angle ,decrease angle,stop x 5 servos and last conncet or not
        self.action_space = Box(low=np.array([-1.0, -1.0,-1.0,-1.0,-1.0,0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.intc)
       
        #Observationspace                          b m1,m2,m3,m4 px,   py  pz c  ilk  pick pos, drop pos 
        self.observation_space = Box(low=np.array([0, 0,0, 0, 0, -0.7,-0.7,-0.7,0,0, -1,-1,-1,-1,-1,-1]),high=np.array([6.2831853072, 2.0943951024, 2.0943951024, 2.0943951024,2.0943951024,0.7,0.7,0.7,1,1,1,1,1,1,1,1]), dtype=np.float32)                     
        # Set start positions
        self.state = PosInit(self.obj_off) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CHECK IF POS REMOVAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Set episod length
        self.episod_length = 1875 #ie 1875 x 32ms = 60 sec   
        
        #setup values for calculating reward      
        self.init_2_pk_dist = get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[10],self.state[11],self.state[12]])
        self.pk_2_dp_dist = get_point_distance([self.state[10],self.state[11],self.state[12]],[self.state[13],self.state[14],self.state[15]])
        
    def step(self,action): 
        
        loadpos = robot.getFromDef("load").getField("translation").getSFVec3f()
               
        
        done =False
        info = {}
        
        #apply action to state variables action
        self.state[0] += (action[0]*0.01745329252)
        self.state[1] += (action[1]*0.01745329252)
        self.state[2] += (action[2]*0.01745329252)
        self.state[3] += (action[3]*0.01745329252)
        self.state[4] += (action[4]*0.01745329252)
        
        if action[5] >0 :
            self.state[8] = 1
        elif action[5]<1 :
            self.state[8] = 0
            
        #Apply state to robot
        updateRobot(self.state)
        
        #Update robot
        robot.step(timestep)
        
        # get observations
        self.state[0]=p0.getValue()
        self.state[1]=p1.getValue()
        self.state[2]=p2.getValue()
        self.state[3]=p3.getValue()
        self.state[4]=p4.getValue()
        
        cpos = gps.getValues()
        
        self.state[5]=cpos[0]
        self.state[6]=cpos[1]
        self.state[7]=cpos[2]
        
        #self.state[8] already applied above while position may or may not be possible ie c
        # note we update get presence after reward for easyness,bcse lock must be called based on previous get presence,
        # since no proper war of detecting connection is possible
        
        # No need to update pick pos and Drop pos
        
               
        #calculate reward
        reward = 0
        #to reduce no of step require , for every step there will be -1
        reward = reward-1
        
        
        # Trying pick up after sucessfull Dropping
        if self.doing == 2 and self.state[8]==1 and self.state[9] == 1:
            reward = -120
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info
        
        #Colliding with load        
        if get_point_distance(loadpos,gps.getValues()) <  self.obj_off:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info
        
        #breaking arm limits
        
        if self.state[0] <= 0 or self.state[0]>=6.2831853072:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info
        
        if self.state[1] <= 0 or self.state[1]>=2.0943951024:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info   
         
        if self.state[2] <= 0 or self.state[2]>=2.0943951024:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info     
        
        if self.state[3] <= 0 or self.state[3]>=2.0943951024:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info  
        
        if self.state[4] <= 0 or self.state[4]>=2.0943951024:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info  
        
        #connector position Y below 0.01
        if self.state[6]<0.01:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info 
        
        #connector position inside base
        if self.state[6]<0.0899 and abs(self.state[5]) < 0.11 and abs(self.state[7]) < 0.11:
            reward-=100
            done = True
            self.state[9]=c1.getPresence()
            return self.state, reward, done, info 
        
        #Dropping in between , note while calculating reward increase y component by height of object
        
        if self.state[8]==0 and self.doing==1 and self.pos_accuracy < get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[13],self.state[14],self.state[15]]):
            reward-=130
            done = True
            return self.state, reward, done, info
        
        #Each unsucessfull pickup tries
        if self.state[8]==1 and self.state[9]==0:
            reward-=10
        
        # Each step penality to reduce time
        reward -=1
        
        
        # positive reward
        
        # returning to initial position after droping or Sucessfull completion
        
        if self.doing == 2 and get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[13],self.state[14],self.state[15]]) > 0.2:
            reward+=120
            self.doing = 3
            done = True
            return self.state, reward, done, info
        
        #Sucessfull pick up
        if self.state[8]==1 and self.state[9]==1 and self.doing == 0:
            reward+=40
            self.doing =1
        
        #Sucessfull Dropping
        if self.doing == 1 and self.state[8] == 0 and self.pos_accuracy > get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[13],self.state[14],self.state[15]]):
            reward+=80
            self.doing =2
        
        
        # decrease episod
        
        self.episod_length -=1  
        #incomplete actions
        
        
        # before picking
        if self.episod_length == 0 and  self.doing == 0:
            Di = self.init_2_pk_dist 
            Df = get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[10],self.state[11],self.state[12]])
            r= ((Di - Df)/Di)*32
            reward+=r            
            done = True
            return self.state, reward, done, info
        
        #After Pickup and Before drop
        if self.episod_length == 0 and  self.doing == 1:
            Di = self.pk_2_dp_dist
            Df = get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[13],self.state[14],self.state[15]])
            r= ((Di - Df)/Di)*48
            reward+=r            
            done = True
            return self.state, reward, done, info     
        
        #After Drop
        if self.episod_length == 0 and  self.doing == 2:
            Di = 0.2
            Df = get_point_distance([self.state[5],self.state[6],self.state[7]],[self.state[13],self.state[14],self.state[15]])
            r= ((Df)/Di)*64 # in this case df increase
            reward+=r            
            done = True
            return self.state, reward, done, info
        
        
 
            
        
        
        
        
        self.state[9]=c1.getPresence() # getPresence() returns -1,0,1 not true or false
        
        
    def render(self):
        pass
    def reset(self):
        pass
     




#while robot.step(timestep) != -1:
    #print()
    
    #pass
    
    
        
        
        
        
        