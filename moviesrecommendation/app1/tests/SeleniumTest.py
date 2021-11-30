from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
import time
import unittest



def Auto():
	driver = webdriver.Chrome()
	driver.get('http://127.0.0.1:8000/')
	driver.set_window_size(1920, 1012)
	time.sleep(5)

	#Going on dropdown and clicking Content based 

	driver.find_element_by_xpath("//a[@id='navbarDropdown']").click()
	time.sleep(1)
	driver.find_element_by_xpath("//body/nav[1]/div[1]/ul[1]/li[4]/div[1]/a[2]").click()
	time.sleep(1)

	#Testing Content based filtering with movie name Iron Man

	driver.find_element_by_name("mname").send_keys("Iron Man")
	time.sleep(1)
	driver.find_element_by_xpath("//button[contains(text(),'Submit')]").click()
	time.sleep(1)

	#Going back to homepage

	driver.back()
	driver.find_element_by_name("mname").clear()
	driver.back()
	time.sleep(1)

	#Going on dropdown and clicking Content Based with KMeans 

	driver.find_element_by_xpath("//a[@id='navbarDropdown']").click()
	time.sleep(1)
	driver.find_element_by_xpath("//a[contains(text(),'Content Based using Kmeans')]").click()
	time.sleep(1)
	
	#Testing Content based filtering with kmeans with movie name Iron Man

	driver.find_element_by_name("mname").send_keys("Iron Man")
	time.sleep(1)
	driver.find_element_by_xpath("//button[contains(text(),'Submit')]").click()
	time.sleep(1)

	#Going back to homepage

	driver.back()
	driver.find_element_by_name("mname").clear()
	driver.back()
	time.sleep(1)

	#Going on dropdown and clicking Collaborative Filtering 

	driver.find_element_by_xpath("//a[@id='navbarDropdown']").click()
	time.sleep(1)
	driver.find_element_by_xpath("//a[contains(text(),'Collaboritive Filtering')]").click()
	time.sleep(1)

	#Testing Collaborative filtering with movie name Iron Man

	driver.find_element_by_name("mname").send_keys("Iron Man")
	time.sleep(1)
	driver.find_element_by_xpath("//button[contains(text(),'Submit')]").click()
	time.sleep(1)

	#Going back to homepage

	driver.back()
	driver.find_element_by_name("mname").clear()
	driver.back()
	time.sleep(1)

	#Going on dropdown and clicking Popularity Based 

	driver.find_element_by_xpath("//a[@id='navbarDropdown']").click()
	time.sleep(1)
	driver.find_element_by_xpath("//a[contains(text(),'Popularity Based')]").click()
	time.sleep(1)

	#Going back to homepage

	#driver.back()
	#driver.find_element_by_name("mname").clear()
	driver.back()
	time.sleep(1)













	driver.close()



Auto()