"""
Alex Casson

Versions
20.04.19 - v1 - initial script

Aim
Check the MSc project allocation

"""

# %% Initalise Python
import pandas as pd
import numpy as np

# %% Load in files
fn_allocation = "allocation_for_checking.xlsx"
allocation = pd.read_excel(fn_allocation,index_col=0)

fn_workload = "staff_workload-16.03.22.csv"
workload = pd.read_csv(fn_workload)
workload.set_index("Supervisor name", inplace=True)

fn_projects = "projects-22.02.22.xlsx"
projects = pd.read_excel(fn_projects,sheet_name="Sheet1",header=0,index_col="ID")

fn_choices = "choices-16.03.22.csv"
choices = pd.read_csv(fn_choices)

fn_courses = "students_and_courses-16.03.22.csv"
course = pd.read_csv(fn_courses)


# %% Check supervisor workload
allocated_load = allocation['Allocated project supervisor'].value_counts().sort_index()
for i in range(len(allocated_load)):
    supervisor = allocated_load.index[i]
    max_load = workload['Ideal workload'].loc[supervisor]
    if allocated_load[i] > max_load:
        print("Error:", supervisor, "is overloaded")



# %% Check number assigned to each project
no_times_project_allocated = allocation['Allocated project ID'].value_counts().sort_index()
for i in range(len(no_times_project_allocated)):
    proj_no = no_times_project_allocated.index[i]
    if proj_no <= len(projects)-1:
        no_times = no_times_project_allocated.loc[proj_no]
        max_times = projects['Please indicate the maximum number of students that your project area would be suitable for.'].loc[proj_no]
        if no_times > max_times:
            print("Error:", proj_no, "allocated more than the maximum number of times")
    else:
        # Bespoke projects after this
        pass

# %% Check student is assigned one of their choices
for i in range(len(choices)):
    if choices['Last Name'].iloc[i] == 'Casson_PreviewUser':
        pass # skip the test user
    else:
        username = choices['Username'].iloc[i]
        answers = (choices['Answer 1'].iloc[i], choices['Answer 2'].iloc[i], choices['Answer 3'].iloc[i], choices['Answer 4'].iloc[i], choices['Answer 5'].iloc[i])
        allocated_project = allocation[allocation['Username'].str.match(username)]
        
        # Check student is assigned one of their choices
        if allocated_project['Choice 1'].iloc[0] == 0: # hard coded 0 as each username will only match once
            pass # student didn't submit choices
        else:
            proj_no = allocated_project['Allocated project ID'].iloc[0] # hard coded 0 as each username will only match once
            project_allocated_in_choices = np.any(answers==proj_no)
            if not project_allocated_in_choices:
                print("Error:", username, "given a project not from their choices")

            # Check that listed course is correct
            course_in_allocation = allocated_project['MSc Programme [Total Pts: 0 Text] |981722'].iloc[0] # hard coded 0 as each username will only match once
            course_in_bb_pd = course[course['Username'].str.match(username)]
            course_in_bb = course_in_bb_pd['MSc Programme [Total Pts: 0 Text] |981722'].iloc[0]
            if not  course_in_allocation == course_in_bb:
                print("Error:", username, "courses do not match up")
            
            
            
# %% Check listed supervisor and title is correct
