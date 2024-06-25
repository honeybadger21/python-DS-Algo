# 2877. Create a DataFrame from List
def createDataframe(student_data: List[List[int]]) -> pd.DataFrame:
    df = pd.DataFrame(student_data)
    df.columns=['student_id', 'age']
    return df

# 2878. Get the Size of a DataFrame
def getDataframeSize(players: pd.DataFrame) -> List[int]:
    return list(players.shape)

