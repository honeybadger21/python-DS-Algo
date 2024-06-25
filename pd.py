# 2877. Create a DataFrame from List
def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
    df = pd.DataFrame(student_data)
    df.columns=['student_id', 'age']
    return df

# 2878. Get the Size of a DataFrame
def getDataframeSize(players: pd.DataFrame) -> List[int]:
    return list(players.shape)

# 2879. Display the First Three Rows
def selectFirstRows(employees: pd.DataFrame) -> pd.DataFrame:
    return employees[:3]

# 2880. Select Data
def selectData(students: pd.DataFrame) -> pd.DataFrame:
    return students[students.student_id == 101][['name', 'age']]

# 2881. Create a New Column
def createBonusColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus'] = employees.salary*2
    return employees

# 2882. Drop Duplicate Rows
def dropDuplicateEmails(customers: pd.DataFrame) -> pd.DataFrame:
    return customers.drop_duplicates(['email'])

# 2883. Drop Missing Data
def dropMissingData(students: pd.DataFrame) -> pd.DataFrame:
    return students.dropna(subset=['name'])

# 2884. Modify Columns
def modifySalaryColumn(employees: pd.DataFrame) -> pd.DataFrame:
    employees['salary'] = employees['salary']*2
    return employees

# 2885. Rename Columns
def renameColumns(students: pd.DataFrame) -> pd.DataFrame:
    students.columns = ['student_id', 'first_name', 'last_name', 'age_in_years']
    return students

# 2886. Change Data Type
def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
    students.grade = students.grade.astype(int)
    return students

# 2887. Fill Missing Data
def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
    products.quantity.fillna(0, inplace=True)
    return products

# 2888. Reshape Data: Concatenate
def concatenateTables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2])











