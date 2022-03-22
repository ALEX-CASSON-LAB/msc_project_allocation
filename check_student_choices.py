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

def check_student_choices(course,projects,student):
    # %% Initalise Python
    import numpy as np
    
    # %% Pre-allocation student checks
    for i in range(len(student)):
        
        # Extract data
        #choices_pd = student[['Choice 1','Choice 2','Choice 3','Choice 4','Choice 5']].iloc[i]
        choices_pd = student[['Answer 1','Answer 2','Answer 3','Answer 4','Answer 5']].iloc[i]
        choices = choices_pd.to_numpy() #pd.DataFrame.to_numpy(choices_pd)
        #choices = pd.DataFrame.to_numpy(pd.to_numeric(choices_pd))
    
        # Check project number is a number
        text = np.array(["".join(item) for item in choices.astype(str)])
        text_check = all(np.char.isnumeric(text[x]) for x in range(len(text)))
        if not text_check:
            print("Error:", student['First Name'].iloc[i], student['Last Name'].iloc[i], "has text in submission, or didn't enter 5 choices. Fix before proceeding")
            #break
    
        # Check project number is in range of project numbers
        if choices.max() > projects.index.max() or choices.min() < projects.index.min():
            print("Error:", student['First Name'].iloc[i], student['Last Name'].iloc[i], "selected a project outside of the valid range. Fix before proceeding")
            continue
        
        # Check project number is integer
        if not np.array_equal(choices_pd, choices_pd.astype(int)):
            print("Error:", student['First Name'].iloc[i], student['Last Name'].iloc[i], "did not enter an integer project number. Fix before proceeding")
            continue
    
        # Check student not submitted same project multiple times
        unique_choices = np.unique(choices)
        if len(unique_choices) != 5:
            print("Warning:", student['First Name'].iloc[i], student['Last Name'].iloc[i], "does not have unique project choices")
    
        # Check projects come from 4 different supervisors
        supervisors = projects['Supervisor name'].loc[choices]
        unique_supervisors = len(supervisors.unique())
        if unique_supervisors < 4:
            print("Warning:", student['First Name'].iloc[i], student['Last Name'].iloc[i], "does not have projects from at least 4 different supervisors")
    
        # Check projects are for correct course
        #programme = student['Programme'].iloc[i]
        programme = student['MSc Programme [Total Pts: 0 Text] |981722'].iloc[i]
        courses = projects['Please select the MSc Programme(s) that your project is most suited to. You may select multiple programmes.'].loc[choices]
        valid_course = courses.str.contains(programme,case=False)
        if not all(valid_course):
            print("Warning:", student['First Name'].iloc[i], student['Last Name'].iloc[i], "has choices not from their course")