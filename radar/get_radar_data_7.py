#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
========================================================================
Purpose: Tracking api
Output: 
		 Position x,y,z 
		 Energy Parameter (sum of matrix)
========================================================================
'''
from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
import time
from sys import platform
from os import system
from imp import load_source
from os.path import join
import scipy.io
import pandas as pd
import datetime 
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation

if platform == 'win32':
	modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
		'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
    modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')     

wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()
#################################
# init globals
#################################
# bedposition dictionary
pos_dict = ['no_person','rueck_mittig','seit_r','seit_l','sitzinbett','bettrandsitz']

# set recording samples
rec_samples = 200  # 1000 == 55sec, 
# choose to record which position	
bedposition = pos_dict[5]
# store data
store_im_mat = False  # mat.file rasterImage: boolean: True/False
store_en = False     # energy: boolean: True/False
store_im = False     # rasterImage: boolean: True/False
# init path
data_path = '/home/studentk/Schreibtisch/projekt_radar/examples/python_self/data_recording/'

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

def PrintSensorTargets(targets):
    system('cls' if platform == 'win32' else 'clear')
    if targets:
        for i, target in enumerate(targets):
             # print pos x,y,z and energy parameter
            print('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                i + 1, target.xPosCm, target.yPosCm, target.zPosCm,
                target.amplitude))
    else:
        print('No Target Detected')

def PrintBreathingEnergy(energy):
    system('cls' if platform == 'win32' else 'clear')
    print('Energy = {}'.format(energy))
    
def save_to_csv(storepath, data):
	''' save data to csv.file via pandas '''
	data_csv = data
	df = pd.DataFrame(data=data_csv)
	#df = pd.DataFrame(columns=data_csv)
	df.to_csv(storepath, mode='a',header=None,index=False)

def SensorApp():
	# wlbt.SetArenaR - input parameters
	minInCm, maxInCm, resInCm = 30, 200, 1
	# wlbt.SetArenaTheta - input parameters
	minIndegrees, maxIndegrees, resIndegrees = -15, 15, 3
	# wlbt.SetArenaPhi - input parameters
	minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -40, 40, 1
	# Set MTI mode
	mtiMode = False
	# Configure Walabot database install location (for windows)
	wlbt.SetSettingsFolder()
	# 1) Connect : Establish communication with walabot.
	wlbt.ConnectAny()
	# 2) Configure: Set scan profile and arena
	# Set Profile - to Sensor.
	wlbt.SetProfile(wlbt.PROF_SENSOR)
	# Setup arena - specify it by Cartesian coordinates.
	wlbt.SetArenaR(minInCm, maxInCm, resInCm)
	# Sets polar range and resolution of arena (parameters in degrees).
	wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
	# Sets azimuth range and resolution of arena.(parameters in degrees).
	wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
	# Moving Target Identification: standard dynamic-imaging filter
	filterType = wlbt.FILTER_TYPE_MTI if mtiMode else wlbt.FILTER_TYPE_NONE
	wlbt.SetDynamicImageFilter(filterType)
	# 3) Start: Start the system in preparation for scanning.
	wlbt.Start()
	if not mtiMode: # if MTI mode is not set - start calibrartion
		# calibrates scanning to ignore or reduce the signals
		wlbt.StartCalibration()
		while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
			wlbt.Trigger()
	###########################################
	# init recording parameter
	###########################################
	# init paths
	path_csv = 'data_recording/data_1.csv'
	path_csv_arena = 'data_recording/data_arena_1.csv'
	path_csv_energy = 'data_recording/data_energy_1.csv'
	path_csv_rasterimage = 'data_recording/data_rasterimage_1.csv'

	# init data recording arrays
	rasterImage_list = []
	image_3D_list = []
	target_list = []
	energy_list = []
	start_data = ['-----start_data_log-----']
	end_data = ['-----end_data_log-----' ]
	
	# save arena params
	data_arena = [timemarker,'rec_samples',rec_samples,'arena_radius',
	minInCm, maxInCm, resInCm,'arena_phi',minPhiInDegrees,maxPhiInDegrees, 
	resPhiInDegrees,'arena_theta',minIndegrees, maxIndegrees, 
	resIndegrees,'mtiMode',mtiMode]
		
	# store data: csv-method
	if store_im is True:
		save_to_csv(path_csv_rasterimage, data_arena)
	
	# init image_list
	imageplot_list = [] # init for animation
	image_list = [] # init for raw data
	
	t1_1 = datetime.datetime.now()
	for sample in range(rec_samples):
		appStatus, calibrationProcess = wlbt.GetStatus()
		# 5) Trigger: Scan(sense) according to profile and record signals
		# to be available for processing and retrieval.
		wlbt.Trigger()
		# 6) Get action: retrieve the last completed triggered recording
		
		# print progress
		if  sample%10 == 0:
			print("Recording samples: "+str(sample)+"/"+str(rec_samples))
	
		###########################################
		# get 2D projection image
		###########################################
		rasterImage, sizeX, sizeY, sliceDepth, power = wlbt.GetRawImageSlice()
	
		#print(sizeX) # size of rasterImage
		#print(sizeY) # size of rasterImage
	
		#~ print(len(rasterImage))
		#~ print(rasterImage)  # e.g. list x=57, with inner lists y=27
		#print(len(rasterImage[-1]))
		#~ print(rasterImage[-1])
		#~ print(len(rasterImage[0]))
	
		#print(sliceDepth)
		#print(power)
		image_2D = np.zeros((sizeX, sizeY))
		#print(image_2D.shape)
		#print(image_2D)
		pos = 0
		for raster in rasterImage:
			#print(raster)
			# fill image_2D
			image_2D[pos,:] = raster
			pos += 1
		#print(image_2D)
		
		# append to list for animation
		#~ imgplot = plt.imshow(image_2D, animated=True)
		#~ imageplot_list.append([imgplot])
		# append to list for data
		image_list.append(image_2D)
		
		###########################################
		# get 3D point cloud data
		###########################################
		#rasterImage3, sizeX3, sizeY3, sizeZ3 , power3 = wlbt.GetRawImage()
		#print(rasterImage3) # list e.g. 7x25x57 (ca. 10,000 data points)
		#print(len(rasterImage3)) # list x
		#print(len(rasterImage3[0])) # list y
		#print(len(rasterImage3[0][0])) # list z
		#print(sizeX3)
		#print(sizeY3)
		#print(sizeZ3)
		#print(power3)
		
		#store data: csv-method
		if store_im is True:
			save_to_csv(path_csv_rasterimage, start_data)
			save_to_csv(path_csv_rasterimage, rasterImage)
			save_to_csv(path_csv_rasterimage, end_data)
		
		###########################################
		# get 3D targets (tracking)
		###########################################
		#~ targets = wlbt.GetSensorTargets()
		#~ # call print out function PrintSensorTargets(targets)
		#~ PrintSensorTargets(targets)
		
		###########################################
		# get Energy
		###########################################
		# 6) Get action: retrieve the last completed triggered recording
		#~ energy = wlbt.GetImageEnergy()
		#~ # PrintBreathingEnergy(energy)
		#~ PrintBreathingEnergy(energy)
		#########################################################
		# get raw data from antenna pair (will raise error!!!)
		#########################################################
		#~ ant_pair = wlbt.GetAntennaPairs()
		#~ print(ant_pair) # list of tuples (antenna 1 bis 15)
		#~ print(len(ant_pair)) # 43 antenna pairs
		#~ print(ant_pair[0]) # AntennaPair(txAntenna=4, rxAntenna=1)
		#~ print(ant_pair[0].txAntenna) # AntennaPair(txAntenna=4, rxAntenna=1)
		#~ print(ant_pair[0].rxAntenna) # AntennaPair(txAntenna=4, rxAntenna=1)
		
		# will raise error!!
		#signal_list, timeAxis_list = wlbt.GetSignal(ant_pair[0])
		
		#########################################################
		# store data in lists
		#########################################################
		#~ energy_list.append(energy)
		#~ rasterImage_list.append(rasterImage)
	
			
	# optional: print out stored data
	#~ print(image_2D_list)
	#~ print(image_3D_list)
	#~ print(target_list)
	#~ print(energy_list)
	#~ print(len(energy_list))
	
	# print out recording time
	t1_2 = datetime.datetime.now()
	rec_time = t1_2-t1_1
	print("total recording time: " + str(rec_time)[0:-7]+'\n')
	
	###########################################
	# start data export 
	###########################################
	#store data: csv-method
	if store_en is True:
		save_to_csv(path_csv_energy, data_arena)
		save_to_csv(path_csv_energy, start_data)
		save_to_csv(path_csv_energy, energy_list)
		save_to_csv(path_csv_energy, end_data)
		#save_to_csv(path_csv_rasterimage, rasterImage_list)
		#save_to_csv(path_csv, rasterImage3)
		print('Data sucessfully stored\n')
				
	# 7) Stop and Disconnect.
	wlbt.Stop()
	wlbt.Disconnect()
	print('Terminate successfully')	
	return image_list, sizeX, sizeY

def video_seq(image_list):
	''' show animation sequence of images'''
	fig = plt.figure()
	im_ani = animation.ArtistAnimation(fig, image_list, interval=50, 
	                                   repeat_delay=1,blit=True)
	# To save this animation with some metadata
	# im_ani.save('im.mp4', metadata={'artist':'mp4'})
	plt.show()
		
def save_matfile(image_list, sizeX, sizeY):
	''' save via .mat file method'''
	data_name = str(timemarker)+"_"+"x"+str(sizeX)+"_"+"y"+str(sizeY)+"_"+"pos_"+bedposition
	
	# reshape list into numpy array
	len_im_list = len(image_list)
	#~ # init image tensor
	#~ image_tensor = np.zeros()
	image_tensor = np.empty(shape=[sizeX,sizeY,len_im_list],dtype=float)
	#print(image_tensor.shape)
	# fill image tensor
	pos = 0
	for image in image_list:
		image_tensor[:,:,pos] = image
		pos += 1	
	# create dict model
	data = {"data_name" : data_name}
	# modelweights
	data["sample_image0"] = image_list[0]
	data["image_tensor"] = image_tensor
	data["num_images"] = len_im_list
	
	print("Data saved to: ")
	print(data_path + data_name)
	scipy.io.savemat(data_path + data_name,data)	
	print('=============================================')
	print("export finished")
	print('=============================================')
	print(" "+"\n")
	
	
if __name__ == '__main__':
	# start sensorApp
	image_list, sizeX, sizeY = SensorApp()
	# save via .mat file method:
	if store_im_mat is True:
		save_matfile(image_list, sizeX, sizeY)
	# start animation
	#video_seq(image_list)
	print("data aquisition finished")
	
	
	
	
	

	
