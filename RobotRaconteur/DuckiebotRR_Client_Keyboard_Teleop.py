from RobotRaconteur.Client import *
import time
import numpy
import sys


def main():
    url='rr+tcp://duckielu:2356?service=Drive'
    if (len(sys.argv)>=2):
        url=sys.argv[1]

    #Instruct Robot Raconteur to use NumPy
    RRN.UseNumPy=True

    #Connect to the service
    c=RRN.ConnectService(url,"cats",{"password":RR.RobotRaconteurVarValue("cats111!","string")})
    input("direction")
    while(1):
        r=input()
        if (r=="q"):
            break
        elif (r=="a"):
            c.setWheelsSpeed(0.5,0.5)
        elif (r=="s"):
            c.setWheelsSpeed(-0.5,-0.5)
        else:
            c.setWheelsSpeed(0,0)


if __name__ == '__main__':
    main()
