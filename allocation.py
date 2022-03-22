"""
Alex Casson

Versions
10.03.22 - v2 - updated with tidy-ups for new academic year
02.04.21 - v1 - initial script

Aim
Perform checks for project allocation and run optimization algorithm

Steps needed:
  - Pre-allocation student checks
    -- Check student not submitted same project multiple times
    -- Check project number is in range and is valid integer
    -- Check the choices come from 4 different supervisors
    -- Check their choices are from correct course

  - Pre-allocation staff checks
    -- Display projects with no selections
    -- Display histogram of popular choices

  - Do allocation

  - Post-allocation student checks 
    -- List number getting 1st/2nd/no choice 

  - Post-allocation staff checks
    -- Display workload

"""


# %% Initalise Python
import pandas as pd
#pd.set_option('display.max_columns', 5) # allow large DataFrames to be displayed
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 # for vecotr fonts in plots
plt.rcParams['ps.fonttype'] = 42 # for vecotr fonts in plots
plt.close('all')
import seaborn as sns
sns.set()
import pulp
import itertools
from getIndexes import getIndexes



# %% Check valid choices from students

# Data files to use
fn_choices = "choices-16.03.22.csv"
fn_courses = "students_and_courses-16.03.22.csv"
fn_projects = "projects-22.02.22.xlsx"

# Load student data
course = pd.read_csv(fn_courses)
projects = pd.read_excel(fn_projects,sheet_name="Sheet1",header=0,index_col="ID")
start_student = pd.read_csv(fn_choices)
student = pd.merge(start_student,course,'inner') # merge so only see students who have submitted choices

# Run checks
from check_student_choices import check_student_choices
check_student_choices(course,projects,student)



# %%

# Load staff workload info
fn_workload = "staff_workload-16.03.22.csv"
workload = pd.read_csv(fn_workload)

# Format workload data frame as required
workload.set_index("Supervisor name", inplace=True)
workload['Workload remaining'] = workload['Ideal workload']



# %% Pre-make the allocation data frame for the final results
student_allocation = student[['Last Name','First Name','Username','MSc Programme [Total Pts: 0 Text] |981722']].copy()
#student_allocation = student[['Last Name','First Name','Username','Programme']]
student_allocation.insert(4,'Allocated project ID',np.nan,True)
student_allocation.insert(5,'Allocated project title',np.nan,True)
student_allocation.insert(6,'Allocated project supervisor',np.nan,True)
student_allocation.insert(7,'Allocated choice no',np.nan,True)
student_allocation.insert(8,'Choice 1',np.nan,True)
student_allocation.insert(9,'Choice 2',np.nan,True)
student_allocation.insert(10,'Choice 3',np.nan,True)
student_allocation.insert(11,'Choice 4',np.nan,True)
student_allocation.insert(12,'Choice 5',np.nan,True)
student_allocation.replace(np.nan, '', regex=True, inplace=True)

#staff_allocation = workload.copy()
staff_allocation = pd.DataFrame()
staff_allocation.insert(0,'Staff name',np.nan,True)
staff_allocation.insert(1,'Allocated project ID',np.nan,True)
staff_allocation.insert(2,'Allocated project title',np.nan,True)
staff_allocation.insert(3,'Allocated username',np.nan,True)
staff_allocation.insert(4,'Allocated last name',np.nan,True)
staff_allocation.insert(5,'Allocated first name',np.nan,True)
staff_allocation.replace(np.nan, '', regex=True, inplace=True)



# %% Pre-allocation staff checks
logfile = "supervisor_log.txt"
print("")
print("Pre-allocation staff checks")
all_choices_pd = student[['Answer 1','Answer 2','Answer 3','Answer 4','Answer 5']]
all_choices_np = pd.DataFrame.to_numpy(all_choices_pd)
all_choices = np.reshape(all_choices_np,np.size(all_choices_np))

# Check for projects not picked by anyone
all_projects = np.arange(1,projects.index.max()+1,1)
all_choices_unique = np.unique(all_choices)
v = np.isin(all_projects,all_choices_unique)
not_chosen_project_number = all_projects[np.invert(v)]
not_chosen_pd = projects[['Supervisor name','Title']].loc[not_chosen_project_number]
print("Projects not chosen")
print(not_chosen_pd)
#not_chosen_pd.to_csv('not_chosen_projects.csv')

# Check for supervisors not picked by any one
chosen_supervisors = projects['Supervisor name'].loc[v]
not_chosen_supervisors = projects['Supervisor name'].loc[np.invert(v)]
chosen_supervisors_unique = chosen_supervisors.unique()
not_chosen_supervisors_unique = not_chosen_supervisors.unique()
w = np.isin(not_chosen_supervisors_unique,chosen_supervisors_unique)
no_project_supervisors = not_chosen_supervisors_unique[np.invert(w)]
print("")
print("Supervisors with no projects chosen", no_project_supervisors)


# Find the 10 most popular choices
counts = np.bincount(all_choices)
counts = np.delete(counts,0)
no_times_chosen = all_projects[np.argsort(-counts)]
top10 = no_times_chosen[0:10]
top10_pd = projects[['Supervisor name','Title']].loc[top10]
print("")
print("Top 10 most popular projects")
print(top10_pd)
#top10_pd.to_csv('top10_chosen_projects.csv')

# Find the 10 most popular, normalized by number of MSc students the project is avaiable to
# TO DO

LOGNOW = True
if LOGNOW:
    
    # Plot number of times each project chosen
    fsize = 20
    fig = plt.hist(all_choices,bins=len(projects))
    xticks = list(range(1,len(projects)+1))
    xticks_labels = projects['Supervisor name'].to_list()
    plt.xticks(xticks, labels=xticks_labels, fontsize=fsize-8)
    plt.xticks(rotation = 90)
    plt.xlabel('Project number', fontsize=fsize)
    #plt.ylabel('Number of times chosen', fontsize=fsize)
    #plt.tight_layout()
    #plt.savefig('supervisor_loading.png',dpi=600)
    
    # Make log file for report
    with open(logfile, 'a') as f:
        print("Supervisors with no projects chosen", no_project_supervisors, file=f)
        print(" ", file=f)
        print("Projects not chosen", file=f)
        f.write(not_chosen_pd.to_string(header = True, index = True))
        print(" ", file=f)
        print("Supervisors with no projects chosen", no_project_supervisors, file=f)
        print(" ", file=f)
        print("Top 10 most popular projects", file=f)
        print(f.write(top10_pd.to_string(header = True, index = True)))




# %% Remove bespoke projects

# Load bespoke project data. This file is made by hand outside of this Python based upon the submitted responses 
fn_bespoke = "bespoke-10.03.22.csv"
bespoke = pd.read_csv(fn_bespoke)

# Remove students with bespoke project from list to allocate
student_copy = student.copy() # backup of whole DataFrame for checking purposes
for i in range(len(bespoke)):
    l = getIndexes(student,bespoke['Username'].iloc[i])
    student.drop([l[0][0]],inplace=True)

# Add students to the list of allocated people
# TODO: Very similar (but not identical!) code is used several times below to update the various lists. Should probably make into some form of function
for i in range(len(bespoke)):
    l = getIndexes(student_allocation,bespoke['Username'].iloc[i])
    m = l[0][0]
    student_allocation.at[m,'Allocated project ID'] = bespoke['Project number'].iloc[i]
    student_allocation.at[m,'Allocated project title'] = bespoke['Title'].iloc[i]
    student_allocation.at[m,'Allocated project supervisor'] = bespoke['Supervisor name'].iloc[i]
    student_allocation.at[m,'Allocated choice no'] = 1
    student_allocation.at[m,'Choice 1'] = 0
    student_allocation.at[m,'Choice 2'] = 0
    student_allocation.at[m,'Choice 3'] = 0
    student_allocation.at[m,'Choice 4'] = 0
    student_allocation.at[m,'Choice 5'] = 0


# Add students to the supervisor list
data = []
for i in range(len(bespoke)):
    a = bespoke['Supervisor name'].iloc[i]
    b = bespoke['Project number'].iloc[i]
    c = bespoke['Title'].iloc[i]
    d = bespoke['Username'].iloc[i]
    e = bespoke['Last Name'].iloc[i]
    f = bespoke['First Name'].iloc[i]
    data.append([a,b,c,d,e,f])
staff_bespoke = pd.DataFrame(data, columns=['Supervisor name', 'Allocated project ID', 'Allocated project title','Allocated username','Allocated last name','Allocated first name'])

# Remove allocation from staff allowed workload
for i in range(len(bespoke)):
    staff_member = bespoke['Supervisor name'].iloc[i]
    workload.at[staff_member,'Workload remaining'] = workload['Workload remaining'].loc[staff_member] - 1
# TODO: Probably good to check for 0 or negative numbers as a safety



# %% Remove 2 year projects which have been allocated by hand 
# Suggest 2 year projects done automatically next time

# Load two year students. This allocation is just done by hand for ease as there aren't many students. 
# They should be given their first or second choice if at all possible, which makes it very similar but slightly different to the main allocation code. Just kept seperate and done by hand for ease.
# Need to make sure (by hand!) that students aren't already in the bespoke project list. Not needed ehre if they are.
fn_two_year_students = "two_year_students-16.03.22.csv"
two_year = pd.read_csv(fn_two_year_students)

# Remove 2 year students from the list of students to allocate as has already been done by hand
two_year = pd.read_csv(fn_two_year_students)
for i in range(len(two_year)):
    l = getIndexes(student,two_year['Username'].iloc[i])
    student.drop([l[0][0]],inplace=True)

# Add these students to the list of allocated students
projects_copy = projects.copy() # Need to drop workload so that project ID is only time a number occurs
projects_copy.drop(columns='Please indicate the maximum number of students that your project area would be suitable for.',inplace=True)
projects_copy.drop(columns='Keyword 1 (e.g. Power systems)',inplace=True)
projects_copy.drop(columns='Keyword 2 (e.g. Artificial intelligence)',inplace=True)
projects_copy.drop(columns='Keyword 3 (e.g. Python)',inplace=True)
projects_copy.drop(columns='Please provide a more detailed description of the topic. Most projects should have 100-300 word descriptions.',inplace=True)
# TODO: This could fall down if there is a number in the project title. Should do a check for this
for i in range(len(two_year)):
    l = getIndexes(student_allocation,two_year['Username'].iloc[i])
    m = l[0][0]
    n = two_year['Allocated project number'].iloc[i]
    student_allocation.at[m,'Allocated project ID'] = two_year['Allocated project number'].iloc[i]
    student_allocation.at[m,'Allocated project title'] = projects_copy['Title'].loc[n]
    student_allocation.at[m,'Allocated project supervisor'] = projects_copy['Supervisor name'].loc[n]
    if two_year['Allocated project number'].iloc[i] == two_year['Answer 1'].iloc[i]:
        student_allocation.at[m,'Allocated choice no'] = 1
    else: # hard coded must be choice 1 or 2, comes from hand allocation but would be better to check
        student_allocation.at[m,'Allocated choice no'] = 2
    student_allocation.at[m,'Choice 1'] = two_year['Answer 1'].iloc[i]
    student_allocation.at[m,'Choice 2'] = two_year['Answer 2'].iloc[i]
    student_allocation.at[m,'Choice 3'] = 0
    student_allocation.at[m,'Choice 4'] = 0
    student_allocation.at[m,'Choice 5'] = 0

# Add students to the supervisor list
data = []
for i in range(len(two_year)):
    n = two_year['Allocated project number'].iloc[i]
    a = projects_copy['Supervisor name'].loc[n]
    b = two_year['Allocated project number'].iloc[i]
    c = projects_copy['Title'].loc[n]
    d = two_year['Username'].iloc[i]
    e = two_year['Last Name'].iloc[i]
    f = two_year['First Name'].iloc[i]
    data.append([a,b,c,d,e,f])
staff_two_year = pd.DataFrame(data, columns=['Staff name', 'Allocated project ID', 'Allocated project title','Allocated username','Allocated last name','Allocated first name'])


# Remove allocation from the staff allowed workload
for i in range(len(two_year)):
    n = two_year['Allocated project number'].iloc[i]
    staff_member = projects_copy['Supervisor name'].loc[n]
    workload.at[staff_member,'Workload remaining'] = workload['Workload remaining'].loc[staff_member] - 1


# Need to also remove from the number of times the project is allowed to be done
for i in range(len(two_year)):
    proj_no = two_year['Allocated project number'].iloc[i]
    projects.at[proj_no,'Please indicate the maximum number of students that your project area would be suitable for.'] = projects['Please indicate the maximum number of students that your project area would be suitable for.'].loc[proj_no] - 1

#TODO: Add explict check for people who are both bespoke and 2 year, is fine this year but could fall through with other choices
if workload['Workload remaining'].min() < 0:
    print("Error: At least one supervisor is over-loaded")


    
# %% Perform main allocation

# Student preference matrix
#weighting = np.array((1, 1, 1, 1, 1)) # equal weighting to all projects
weighting = np.array((1, 1.1, 1.4, 2, 3)) # custom weighting, probably to try and give more people their 1st choice
preference_matrix = np.zeros((len(student),len(projects)))
for i in range(len(student)): # switch to 1, 2, 3, 4 for more staightforwards weighting
    preference_matrix[i,student['Answer 1'].iloc[i]-1] = 1 * weighting [0] # 1 # -1 as zero indexed in numpy, but 1 indexed in Excel
    preference_matrix[i,student['Answer 2'].iloc[i]-1] = 1 * weighting [1] # 2
    preference_matrix[i,student['Answer 3'].iloc[i]-1] = 1 * weighting [2] # 3
    preference_matrix[i,student['Answer 4'].iloc[i]-1] = 1 * weighting [3] # 4
    preference_matrix[i,student['Answer 5'].iloc[i]-1] = 1 * weighting [4] # 5
preference_matrix = preference_matrix # Comment/uncomment depending on whether choice number is included in weight
#preference_matrix[preference_matrix >= 1] = 1 # Comment/uncomment depending on whether choice number is included in weight


# Supervisor matrix
supervisor_matrix = np.zeros((len(workload),len(projects)),dtype=int)
for j in range(len(projects)):
    sup = projects['Supervisor name'].loc[j+1] # +1 as zero indexed in numpy, but 1 indexed in Excel
    idx = workload['Supervisor number'].loc[sup]
    supervisor_matrix[idx-1,j] = 1 # -1 as zero indexed in numpy, but 1 indexed in Excel
#S = np.ones((len(workload),len(projects)))
#S=S.astype(bool)


# Define problem inputs
A = preference_matrix.tolist() # student preferences
chi = ((projects['Please indicate the maximum number of students that your project area would be suitable for.']).to_numpy()).tolist() # max number of students per project
beta = ((workload['Workload remaining']).to_numpy()).tolist() # max workload per staff member
number_of_students, number_of_projects = preference_matrix.shape
number_of_supervisors = len(beta)
S = supervisor_matrix.tolist() 


# Set up problem in Pulp
prob = pulp.LpProblem("project_allocation", pulp.LpMaximize)
x = pulp.LpVariable.dicts("x", itertools.product(range(number_of_students), range(number_of_projects)), cat=pulp.LpBinary)  # Variables


# Set main objective function, want to maximize x with downweighting for higher number (less preferable) choices
objective_function1 = 0
for stud in range(number_of_students):
    for proj in range(number_of_projects):
        if A[stud][proj] > 0:
            objective_function1 += x[(stud, proj)] * 1 / (A[stud][proj])
#prob += objective_function1

# Set second objective function, maximise the number of different supervisors used to spread load
superv_loading_protoypte = np.zeros((number_of_students,number_of_projects),dtype=int)
superv_loading = superv_loading_protoypte.tolist()
superv_loading_this_supervisor_protypte = np.zeros((number_of_supervisors),dtype=int)
superv_loading_this_supervisor = superv_loading_this_supervisor_protypte.tolist()
for k, superv in enumerate(S):
    #print(k)
    for stud in range(number_of_students):
        for proj in range(number_of_projects):
                superv_loading[stud][proj] = workload['Weighting'].iloc[k] * superv[proj] * pulp.lpSum(x[(stud, proj)])
    superv_loading_this_supervisor[k] = pulp.lpSum(superv_loading)
    #print(superv_loading_this_supervisor[k])
objective_function2 = pulp.lpSum(superv_loading_this_supervisor)


# Overall objective function
#weight = 0.3 # in principle could alter the weighting between meeting student preferences and distributing staff loading (not used at present, both equally weighted)
objective_function = objective_function1 + objective_function2
prob += objective_function


# Add constraints
# Students can only be allocated to projects they have chosen
for stud, proj in x:
    #prob += x[stud, proj] <= float(A[stud, proj])
    prob += x[stud, proj] <= float(A[stud][proj])
    
# At most chi_j studdent for every project
for proj in range(number_of_projects):
    prob += pulp.lpSum(x[(stud, proj)] for stud in range(number_of_students)) <= chi[proj]

# At most 1 project per stududent
for stud in range(number_of_students):
    prob += pulp.lpSum(x[(stud, proj)] for proj in range(number_of_projects)) <= 1

# Limit workload per supervisor
for k, superv in enumerate(S):
    supervisor_load = pulp.lpSum(superv[proj] * pulp.lpSum(x[(stud, proj)] for stud in range(number_of_students)) for proj in range(number_of_projects))
    prob += supervisor_load <= beta[k]  


## Find solution
#solver = pulp.PULP_CBC_CMD()
#solver.keepFiles = True
#status = prob.solve(solver) # save lot files version
status = prob.solve()
if status != 1:
    print("Error: No solution to allocation found")
solution = np.array([[ (x.get((i, j))).value() for i in range(number_of_students)] for j in range(number_of_projects)] )
solution = solution.T


# Extract allocated projects and assign NaN if not allocated
allocation = (np.vstack((np.arange(len(solution)),np.full([len(solution)], np.nan)))).T
e = np.nonzero(solution)
allocation[e[0],1] = e[1]
allocated_project = (allocation[:,1] + 1) # Correct for zero indexing in Python

# Add these students to the list of allocated students
# TODO: This could fall down if there is a number in the project title. Should do a check for this
student.reset_index(drop=True,inplace=True)
for i in range(len(student)):
    if np.isnan(allocated_project[i]): # not allocated a project, so skip
        pass
    else:
        l = getIndexes(student_allocation,student['Username'].iloc[i])
        m = l[0][0]
        proj_no = allocated_project[i]
        student_allocation.at[m,'Allocated project ID'] = proj_no
        student_allocation.at[m,'Allocated project title'] = projects_copy['Title'].loc[proj_no]
        student_allocation.at[m,'Allocated project supervisor'] = projects_copy['Supervisor name'].loc[proj_no]
        if proj_no == student['Answer 1'].iloc[i]:
            student_allocation.at[m,'Allocated choice no'] = 1
        elif proj_no == student['Answer 2'].iloc[i]:
            student_allocation.at[m,'Allocated choice no'] = 2
        elif proj_no == student['Answer 3'].iloc[i]:
            student_allocation.at[m,'Allocated choice no'] = 3
        elif proj_no == student['Answer 4'].iloc[i]:
            student_allocation.at[m,'Allocated choice no'] = 4
        elif proj_no == student['Answer 5'].iloc[i]:
            student_allocation.at[m,'Allocated choice no'] = 5
        else:
            print("Error: Not able to identify which choice student was given",i)
        student_allocation.at[m,'Choice 1'] = student['Answer 1'].iloc[i]
        student_allocation.at[m,'Choice 2'] = student['Answer 2'].iloc[i]
        student_allocation.at[m,'Choice 3'] = student['Answer 3'].iloc[i]
        student_allocation.at[m,'Choice 4'] = student['Answer 4'].iloc[i]
        student_allocation.at[m,'Choice 5'] = student['Answer 5'].iloc[i]

# Check the distribution to make sure happy with numbers get 1st, 2nd, 3rd choice etc. If not, can adjust optimization weights and rerun
#student_allocation['Allocated choice no'].value_counts().sort_index().plot(kind='bar')


# Add students to the supervisor list
data = []
for i in range(len(student)):
    if np.isnan(allocated_project[i]): # not allocated a project, so skip
        pass
    else:
        n = allocated_project[i]
        a = projects_copy['Supervisor name'].loc[n]
        b = n
        c = projects_copy['Title'].loc[n]
        d = student['Username'].iloc[i]
        e = student['Last Name'].iloc[i]
        f = student['First Name'].iloc[i]
        data.append([a,b,c,d,e,f])
staff_one_year = pd.DataFrame(data, columns=['Staff name', 'Allocated project ID', 'Allocated project title','Allocated username','Allocated last name','Allocated first name'])


# Remove allocation from the staff allowed workload
for i in range(len(student)):
    if np.isnan(allocated_project[i]): # not allocated a project, so skip
        pass
    else:
        n = allocated_project[i]
        staff_member = projects_copy['Supervisor name'].loc[n]
        workload.at[staff_member,'Workload remaining'] = workload['Workload remaining'].loc[staff_member] - 1



# %% FInd projects for the not-allocated students by hand

# Make list of not allocated students to be handled seperately
not_allocated_tp = np.where(np.isnan(allocated_project))
not_allocated = not_allocated_tp[0]
print("")
print("Warning: There were", len(not_allocated), "students not allocated a project")
for i in range(len(not_allocated)):
    print("    Warning:", student['First Name'].iloc[not_allocated[i]], student['Last Name'].iloc[not_allocated[i]], "was not allocated a project")
#allocated_project_no = (np.delete(allocated_project,not_allocated)).astype(int)

# Make a blank DataFrame matching the wanted format, with the student names in.
# This will be saved as an external file, allocated by hand and then re-loaded back in
# Reduced workload of people before the allocation, so can just take back up
# Could automate more next year
data = []
for i in range(len(not_allocated)):
    a = student['Last Name'].iloc[not_allocated[i]]
    b = student['First Name'].iloc[not_allocated[i]]
    c = student['Username'].iloc[not_allocated[i]]
    d = student['MSc Programme [Total Pts: 0 Text] |981722'].iloc[not_allocated[i]]
    e = 0 # Allocated project ID
    f = 'None' # Allocated project title
    g = 'None' # Allocated project supervisor
    h = 0 # Allocated choice no
    j = student['Answer 1'].iloc[not_allocated[i]]
    k = student['Answer 2'].iloc[not_allocated[i]]
    l = student['Answer 3'].iloc[not_allocated[i]]
    m = student['Answer 4'].iloc[not_allocated[i]]
    n = student['Answer 5'].iloc[not_allocated[i]]
    data.append([a,b,c,d,e,f,g,h,j,k,l,m,n])
to_allocate_by_hand = pd.DataFrame(data,columns = list(student_allocation.columns))


HANDNOW = False
if HANDNOW:
    to_allocate_by_hand.to_excel("to_allocate_by_hand.xlsx")
    # This is written out, the allocation done by hand by checking against the max load each person can have and added to the main projects spreadsheet to be reloaded in


# Code for checking the hand allocation meets the rules
#x = 13
#print("Project no: ", x)
#print("No times already allocated: ", np.sum(allocated_project == x), 'Max allowed number: ', projects['Maximum number of students'].loc[x])
#supername = projects['Supervisor name'].loc[x]
#print("Supervisor: ", supername, 'Ideal workload: ', workload['Ideal workload'].loc[supername], 'Max workload: ', workload['Absolute max workload'].loc[supername])


# # %%
# # Load the spreadsheet back in and add to the main allocation table
# hand_allocated = pd.read_excel(fn_staff,sheet_name="Hand allocation info",index_col=0)
# for i in range(len(hand_allocated)):
#     l = getIndexes(student_allocation,hand_allocated['Username'].iloc[i])
#     m = l[0][0]
#     proj_no = hand_allocated['Allocated project ID'].iloc[i]
#     student_allocation.at[m,'Allocated project ID'] = proj_no
#     student_allocation.at[m,'Allocated project title'] = projects_copy['Title'].loc[proj_no]
#     student_allocation.at[m,'Allocated project supervisor'] = projects_copy['Supervisor name'].loc[proj_no]
#     student_allocation.at[m,'Allocated choice no'] = hand_allocated['Allocated choice no'].iloc[i]
#     student_allocation.at[m,'Choice 1'] = hand_allocated['Choice 1'].iloc[i]
#     student_allocation.at[m,'Choice 2'] = hand_allocated['Choice 2'].iloc[i]
#     student_allocation.at[m,'Choice 3'] = hand_allocated['Choice 3'].iloc[i]
#     student_allocation.at[m,'Choice 4'] = hand_allocated['Choice 4'].iloc[i]
#     student_allocation.at[m,'Choice 5'] = hand_allocated['Choice 5'].iloc[i]


# # Add students to the supervisor list
# data = []
# for i in range(len(hand_allocated)):
#     proj_no = hand_allocated['Allocated project ID'].iloc[i]  
#     a = projects_copy['Supervisor name'].loc[proj_no]
#     b = proj_no
#     c = projects_copy['Title'].loc[proj_no]
#     d = hand_allocated['Username'].iloc[i]
#     e = hand_allocated['Last Name'].iloc[i]
#     f = hand_allocated['First Name'].iloc[i]
#     data.append([a,b,c,d,e,f])
# staff_hand = pd.DataFrame(data, columns=['Staff name', 'Allocated project ID', 'Allocated project title','Allocated username','Allocated last name','Allocated first name'])


# %% Start of post allocation checks

# Check number of students getting first, second etc choice
# Done here as to not include those people who didn't submit a choice
# Staff numbers are provisional until do people who didn't submit
STATSNOW = False
if STATSNOW:
    # Print student numbers getting each choice
    student_allocation['Allocated choice no'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Allocated choice no')
    plt.ylabel('Number of students')
    
    # # Number of projects per staff member
    
    # plt.figure()
    # staff['Staff name'].value_counts().sort_index().plot(kind='bar')
    # plt.ylabel('Number of students')
    
    # # Find staff not allocated projects
    # for i in range(len(workload)):
    #     snames = workload.index
    #     sname = snames[i]
    #     in_list = staff['Staff name'].str.find(sname)
    #     in_list = in_list + 1



# %% Allocate students who did not submit a choice. Done randomly/by hand to supervisors who are underloaded but suitable for their course

# Make complete staff allocation list
#staff = pd.concat((staff_allocation, staff_bespoke, staff_two_year, staff_one_year, staff_hand))
staff = pd.concat((staff_allocation, staff_bespoke, staff_two_year, staff_one_year))
staff.set_index('Staff name',inplace=True)
idx1 = workload.index
idx2 = staff.index
staff_with_no_projects = idx1.difference(idx2)
print("")
print("Warning: The following staff have no project allocation: ")
print(staff_with_no_projects)
print("")

# Identify students who did not submit any choices
idx3 = course.set_index('Username').index
idx4 = student_allocation.set_index('Username').index
students_with_no_choices = idx3.difference(idx4)
print("Warning:", len(students_with_no_choices), "students did not submit choices")
print(students_with_no_choices)

# Load the spreadsheet back in and add to the main allocation table
fn_no_choice_submitted = "students_who_didnt_submit_choices-17.03.22.csv"
no_choice_submitted = pd.read_csv(fn_no_choice_submitted)
data = []
for i in range(len(no_choice_submitted)):
    l = getIndexes(course,no_choice_submitted['Username'].iloc[i])
    m = l[0][0]
    proj_no = no_choice_submitted['Allocated project ID'].iloc[i]
    a = course['Last Name'].iloc[m]
    b = course['First Name'].iloc[m]
    c = course['Username'].iloc[m]
    e = course['MSc Programme [Total Pts: 0 Text] |981722'].iloc[m]
    f = proj_no
    g = projects_copy['Title'].loc[proj_no]
    h = projects_copy['Supervisor name'].loc[proj_no]
    j = 0
    data.append([a,b,c,e,f,g,h,j,j,j,j,j,j])
not_chosen_allocation = pd.DataFrame(data,columns = list(student_allocation.columns))    
student_final = pd.concat([student_allocation, not_chosen_allocation],ignore_index=True)
#student_final = student_allocation.append(not_chosen_allocation,ignore_index=True)    
#student_final.reset_index(drop=True,inplace=True)


# Add students to the supervisor list
data = []
for i in range(len(no_choice_submitted)):
    l = getIndexes(course,no_choice_submitted['Username'].iloc[i])
    m = l[0][0]
    proj_no = no_choice_submitted['Allocated project ID'].iloc[i] 
    a = projects_copy['Supervisor name'].loc[proj_no]
    b = proj_no
    c = projects_copy['Title'].loc[proj_no]
    d = course['Username'].iloc[m]
    e = course['Last Name'].iloc[m]
    f = course['First Name'].iloc[m]
    data.append([a,b,c,d,e,f])
staff_not_chosen = pd.DataFrame(data, columns=['Staff name', 'Allocated project ID', 'Allocated project title','Allocated username','Allocated last name','Allocated first name'])
staff_not_chosen.set_index('Staff name',inplace=True)
staff_final = pd.concat([staff, staff_not_chosen])
#staff_final = staff.append(staff_not_chosen)    


# # %% Manual corrections after main allocation
# def updateAllocation(update,student_final):
#     l = getIndexes(student_final,update[0])
#     m = l[0][0]
#     proj_no = update[1]
#     student_final.at[m,'Allocated project ID'] = proj_no
#     student_final.at[m,'Allocated project title'] = projects_copy['Title'].loc[proj_no]
#     student_final.at[m,'Allocated project supervisor'] = projects_copy['Supervisor name'].loc[proj_no]
#     student_final.at[m,'Allocated choice no'] = update[2]
#     return

# # Assumes workload and project numbers checked by hand - doesn't update numbers here
# # Also doesn't update the supervisor matrix - assumes not going to use this
# update1 = ('cc4323cc',108,2) # username to update, new project number, choice no of new project
# updateAllocation(update1,student_final)
# update2 = ('dd4323dd',127,4) # username to update, new project number, choice no of new project
# updateAllocation(update2,student_final)


# %% Final allocation stats

# Plot student numbers getting each choice
plt.figure()
student_final['Allocated choice no'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Allocated choice no')
plt.ylabel('Number of students')
plt.axis([0.5, 5.5, 0, 100]) # don't display 0 values which are people who didn't submit
#plt.savefig('student_allocation.png',dpi=600)

# Plot supervisor loads
plt.figure()
axis = student_final['Allocated project supervisor'].value_counts().plot(kind='bar')
fig = axis.get_figure()
sns.set(font_scale=1.5)
for tick in axis.get_xticklabels():
    tick.set_rotation(90)
axis.set_ylabel("Number of students")
plt.tight_layout()
#plt.savefig('supervisor_loading.png',dpi=600)

# Plot supervisor under-load
plt.figure()
axis2 = workload['Workload remaining'].plot(kind='bar')
fig = axis2.get_figure()
sns.set(font_scale=1.5)
for tick in axis2.get_xticklabels():
    tick.set_rotation(90)
axis2.set_ylabel("Number of students")
plt.tight_layout()
#plt.savefig('supervisor_underloading.png',dpi=600)

# %% Prepare for saving out

# TODO: Do this earlier so isn't needed here!
# Swap order of columns for easier reading of output
cols = student_final.columns.tolist()
cols[4], cols[6] = cols[6], cols[4]
student_final = student_final.reindex(columns=cols)

# Write out
WRITEWNOW = True
if WRITEWNOW:
    student_final.to_excel("allocation_for_checking.xlsx")