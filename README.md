# msc_project_allocation
Code for allocating student projects to supervisors based on the available projects, submitted choices, and staff workload. 

In the Department of Electrical and Electronic Engineering at the University of Manchester we run 4 MSc courses (ACSE, CASP, EPSE and REACT) each of which can be taken as a one year or two year (ACSE_wER, CASP_wER, AEPSE, REACT_wER) option. Each student must be allocated a project supervisor. The code used for this is here.

# Problem statement
Each student is asked to submit project choices from a list given to them. The list of projects is in the projects.xlsx file (it is formatted a bit more nicely for students). They must each submit 5 different project numbers, from at least 4 different supervisors. In addition students may submit a bespoke project proposal, if they get prior approval from a supervisor. 

Here we need to ensure that each student has submitted valid choices (5 different ones, from 4 supervisors, only for projects suitable for their course) and then allocate the students to projects. Accepted bespoke projects are allocated first. Then two year students are allocated, and should be given their first or second choice. One year students are then allocated and can be given any of their choices (with a slight weighting towards higher numbered choices, but the communications are very clear they could be given any of their choices). Projects also need to be given to students who didn't submit any choices before the deadline. In addition to the student choices each project can potentially be done by my than one person (a column in the projects.xlsx file) but each staff member also has a maximum number that they can supervise (in staff_workload.csv) based on their other workload.

# Code basis
The code is based upon the blog post at https://vknight.org/unpeudemath/math/2017/09/26/allocating-final-year-projects-to-students.html, with some customisations for our needs. In general the blog post is very good and describes the linear algebra approach taken. I have added terms to the optimization to try and maximise the number of different staff given projects to supervise (in order to spread the staff workload) and to allow individual staff members to be up-weighted to make their projects more likely to be chosen (again to spread the load).

# Environment set up
This code is written in Python. The spec-file.txt file contains the required environment information allowing you to load in the required libraries. Instructions on how to re-create the required environment are included in that file.

# Files present and running the code
The main script to run is allocation.py. This calls check_student_choices.py and getIndexes.py. allocation.py isn't really intended as a batch file. You probably want to run it a cell at a time, fix any issues, and then move on to the next cell. Various parts of the allocation are done by hand, and it's a slightly iterative process. 

check.py can be run after the allocation has been completed. It runs an independent check that all of the requirements have been satisfied (students have one of their choices, staff aren't overloaded, and so on).

There are then a number of tables, mainly in .csv format, that are loaded in with the required information. The formatting/column names in here are given by the software we have (Blackboard and Microsoft Forms). 

# This code
For this public version all of the names and projects have been anonymised. Project descriptions were generated using the =RAND() function in Word and supervisor names using the name generator at https://commentpicker.com/fake-name-generator.php. You'll get a lot of "Warning: Firstname 264 Lastname 264 has choices not from their course" warnings as everything is randomly generated - I've not made the data source a valid one! If this was a real data set you'd have to go back to the students and get valid choices before continuing. 

# Limitations
 * The Pandas frame update code is very repetitive. It really should be put into a function rather than repeated multiple times.
 * The column names would benefit from being defined up-front and then called. At the moment they are hard-coded. If these change for any reason you need to be careful to make sure you change all of them. 
