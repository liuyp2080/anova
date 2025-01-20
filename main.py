from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pingouin as pg
import json
from typing import List, Optional

app = FastAPI()

# t test need value and group, anova need value and group, repeated measures anova need value, time, and subject
class RepeatedMeasuresAnovaInput(BaseModel):
    value_column: List[int]
    group_column: Optional[List[object]]=None
    time_column: Optional[List[object]]=None
    subject_column: Optional[List[int]] = None
    paired: Optional[bool] = False

#--------------------anova or t test-------------------------------------
@app.post("/normality_test")
def normality_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """Perform a normality test."""
    df = pd.DataFrame({"values": input_data.value_column, "groups": input_data.group_column})
    results = pg.normality(df, dv="values", group="groups")
    output = results.to_dict(orient="records")
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# homoscedasticity test,  
@app.post("/homoscedasticity_test")
def homoscedasticity_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """Perform a homoscedasticity test."""
    df = pd.DataFrame({"values": input_data.value_column, "groups": input_data.group_column})
    results = pg.homoscedasticity(df, dv="values", group="groups")
    output = results.to_dict(orient="records")
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# if the homogenity test is passed, then the anova test can be performed, suitable for data with more than 2 groups
@app.post("/anova")
def perform_anova(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Perform an ANOVA test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput object containing value and group columns.

    Returns:
        A JSON string containing the ANOVA test results.
    """
    df = pd.DataFrame({
        'value': input_data.value_column,
        'group': input_data.group_column
    })
    anova_results = pg.anova(dv='value', between='group', data=df, detailed=True)
    output = anova_results.to_dict(orient="records")
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# corresponding  pairwise test
@app.post("/pairwise_test_tukey")
def perform_pairwise_tukey_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """Perform Tukey's pairwise test on the input data."""
    df = pd.DataFrame({
        'value': input_data.value_column,
        'group': input_data.group_column
    })
    tukey_results = pg.pairwise_tukey(dv='value', between='group', data=df)
    return json.dumps(tukey_results.to_dict(orient="records"), allow_nan=True, ensure_ascii=False)
# if the homogenity test is not passed, then the welch anova test can be performed
@app.post("/welch_anova")
def perform_welch_anova(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Perform Welch's ANOVA test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput object containing value and group columns.

    Returns:
        A JSON string containing the Welch's ANOVA test results.
    """
    data_frame = pd.DataFrame({
        'value': input_data.value_column,
        'group': input_data.group_column
    })

    welch_anova_results = pg.welch_anova(dv='value', between='group', data=data_frame)
    output = welch_anova_results.to_dict(orient="records")
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# corresponding  pairwise test 
@app.post("/pairwise_test_gameshowell")
def perform_pairwise_gameshowell_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Perform Games-Howell pairwise test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput object containing value and group columns.

    Returns:
        A JSON string containing the Games-Howell pairwise test results.
    """
    data_frame = pd.DataFrame({"value": input_data.value_column, "group": input_data.group_column})
    results = pg.pairwise_gameshowell(dv="value", between="group", data=data_frame)
    output = results.to_dict(orient="records")
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# power of anova
@app.post("/power_anova")
def calculate_power(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Calculate the power of an ANOVA test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput object containing value and group columns.

    Returns:
        A JSON string containing the power of the ANOVA test and the sample size required for 80% power.
    """
    data_frame = pd.DataFrame({"value": input_data.value_column, "group": input_data.group_column})

    # Calculate the power of the ANOVA test
    anova_results = pg.anova(dv="value", between="group", data=data_frame)
    eta_squared = anova_results["np2"][0]
    k = data_frame["group"].nunique()
    n = data_frame["value"].count()
    power = pg.power_anova(eta_squared=eta_squared, k=k, n=n)

    # Calculate the sample size required for 80% power
    n_80 = pg.power_anova(eta_squared=eta_squared, k=k, power=0.8)

    # Create the output dictionary
    output = {
        "power_anova": power,
        "n_80": n_80
    }

    # Return the JSON string
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# -------------------if the group is binary, use t test or man-whitney u test-------------------
@app.post("/t_test")
def t_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Perform a t-test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput object containing value and group columns.

    Returns:
        A JSON string containing the results of the t-test.
    """
    data_frame = pd.DataFrame({"values": input_data.value_column, "groups": input_data.group_column})

    group_names = data_frame["groups"].unique()
    group1_values = data_frame[data_frame["groups"] == group_names[0]]["values"]
    group2_values = data_frame[data_frame["groups"] == group_names[1]]["values"]

    t_test_results = pg.ttest(
        x=group1_values,
        y=group2_values,
        paired=input_data.paired
    )
    t_test_results['CI95%']=[ value.tolist() for value in t_test_results['CI95%'] ]

    output = t_test_results.to_dict(orient="records")

    return json.dumps(output, allow_nan=True, ensure_ascii=False)

@app.post("/mann_whitney_u_test")
def perform_mann_whitney_u_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Perform a Mann-Whitney U test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput

    Returns:
        A JSON string containing the results of the test.
    """
    dataframe = pd.DataFrame({
        "values": input_data.value_column,
        "groups": input_data.group_column
    })

    group_names = dataframe["groups"].unique()
    group1_values = dataframe[dataframe["groups"] == group_names[0]]["values"]
    group2_values = dataframe[dataframe["groups"] == group_names[1]]["values"]

    results = pg.mwu(x=group1_values, y=group2_values)
    output = results.to_dict(orient="records")

    return json.dumps(output, allow_nan=True, ensure_ascii=False)
# paired non-parametric test
@app.post("/wilcoxon_signed_rank_test")
def wilcoxon_signed_rank_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """
    Perform a Wilcoxon signed-rank test on the input data.

    Args:
        input_data: RepeatedMeasuresAnovaInput

    Returns:
        A JSON string containing the results of the test.
    """
    df = pd.DataFrame({"values": input_data.value_column, "groups": input_data.group_column})

    group_names = df["groups"].unique()
    group1_values = df[df["groups"] == group_names[0]]["values"]
    group2_values = df[df["groups"] == group_names[1]]["values"]

    results = pg.wilcoxon(group1_values, group2_values, alternative="two-sided")
    output = results.to_dict(orient="records")

    return json.dumps(output, allow_nan=True, ensure_ascii=False)

# power of t test
@app.post("/power_t_test")
def power_t_test(input_data: RepeatedMeasuresAnovaInput) -> str:
    """Calculate the power of a t-test and the sample size needed for 80% power."""
    
    df = pd.DataFrame({
        'value': input_data.value_column,
        'group': input_data.group_column
    })
    
    groups = df['group'].unique()
    group1_values = df[df['group'] == groups[0]]['value']
    group2_values = df[df['group'] == groups[1]]['value']
    
    t_test_results = pg.ttest(x=group1_values, y=group2_values, paired=input_data.paired)
    
    power_of_test = t_test_results['power'].iloc[0]
    sample_size_80_power = pg.power_ttest(power=0.8,d=t_test_results['cohen-d'].iloc[0])
    
    output = {
        'power_ttest': power_of_test,
        'n_80': sample_size_80_power
    }
    
    return json.dumps(output, allow_nan=True, ensure_ascii=False)


#--------------------repreat measures anova-----------------------------
@app.post("/sphericity_test")
def sphericity_test(input_data: RepeatedMeasuresAnovaInput):
    """
    Perform a Mauchly's test of sphericity.

    Args:
        input_data: RepeatedMeasuresAnovaInput

    Returns:
        A JSON string containing the results of the sphericity test
    """
    data_frame = pd.DataFrame({
        'value': input_data.value_column,
        'time': input_data.time_column,
        'subject': input_data.subject_column
    })
    try:
        spher, W, chisq, dof, pval  = pg.sphericity(data=data_frame, dv='value', within='time', subject='subject')
        output = {
            'sphericity': spher,
            'W': W,
            'chisq': chisq,
            'dof': dof,
            'pval': pval
        }
        return json.dumps(output, allow_nan=True,ensure_ascii=False)
    except Exception as e:
        return str(e)

@app.post("/epsilon_correct_factor")
def epsilon_correct_factor(input_data: RepeatedMeasuresAnovaInput):
    """
    Calculate epsilon correction factors (Greenhouse-Geisser, Huynh-Feldt, and Lower-bound)
    for repeated measures ANOVA.

    Args:
        input_data: RepeatedMeasuresAnovaInput containing columns for values, time points, and subjects.

    Returns:
        A JSON string with epsilon correction factors:
        - 'gg': Greenhouse-Geisser correction
        - 'hf': Huynh-Feldt correction
        - 'lb': Lower-bound correction
    """

    data_frame = pd.DataFrame({
        'value': input_data.value_column,
        'time': input_data.time_column,
        'subject': input_data.subject_column
    })

    gg = pg.epsilon(data=data_frame, dv='value', within='time', subject='subject',correction='gg')
    hf = pg.epsilon(data=data_frame, dv='value', within='time', subject='subject',correction='hf')
    lb= pg.epsilon(data=data_frame, dv='value', within='time', subject='subject',correction='lb')

    result = {
        'gg': gg,
        'hf': hf,
        'lb': lb
    }
    
    return json.dumps(result, allow_nan=True,ensure_ascii=False)

@app.post("/two_way_repeated_measures_anova")
def two_way_repeated_measures_anova(input_data: RepeatedMeasuresAnovaInput):
    """
    Perform a repeated measures ANOVA. if the group column is not None, perform a mixed ANOVA, else perform a repeated measures ANOVA

    Args:
        input_data: RepeatedMeasuresAnovaInput

    Returns:
        A JSON string containing the results of the ANOVA
    """
    df = pd.DataFrame({
        'value': input_data.value_column,
        'time': input_data.time_column,
        'subject': input_data.subject_column
    })
    
    if input_data.group_column is not None:
        df['group'] = input_data.group_column
        results = pg.mixed_anova(
                data=df,
                dv='value',
                within='time',
                between='group',
                subject='subject',
                correction= True
            )
        output = results.to_dict(orient="records")
    else:
        results = pg.rm_anova(
                data=df,
                dv='value',
                within='time',
                subject='subject',
                correction= True
            )
        output = results.to_dict(orient="records")

    return json.dumps(output, allow_nan=True,ensure_ascii=False)

#paired_test
@app.post("/paired_test")
def pairwise_test(input_data: RepeatedMeasuresAnovaInput):
    """
    Perform a pairwise test (e.g. Wilcoxon signed-rank test) between each pair of time points.

    Args:
        input_data: RepeatedMeasuresAnovaInput

    Returns:
        A JSON string containing the results of the pairwise tests
    """
    data_frame = pd.DataFrame({
        'value': input_data.value_column,
        'time': input_data.time_column,
        'subject': input_data.subject_column
    })
    if input_data.group_column is not None:
        data_frame['group'] = input_data.group_column
        
        paired_test_results = pg.pairwise_tests(
            data=data_frame,
            dv='value',
            within='time',
            between='group',
            subject='subject'
        )
    else:
        paired_test_results = pg.pairwise_tests(
            data=data_frame,
            dv='value',
            within='time',
            subject='subject'
        )

    output = paired_test_results.to_dict(orient="records")
    return json.dumps(output, allow_nan=True,ensure_ascii=False)
@app.post("/power_rm_anova")
def power_rm_anova(input_data: RepeatedMeasuresAnovaInput) -> str:
    """Perform power analysis for a repeated measures ANOVA.

    Args:
        input_data: RepeatedMeasuresAnovaInput containing columns for values, time points, and subjects.

    Returns:
        A JSON string containing the results of the power analysis:
        - 'power': the power of the test
        - 'n at 80% power': the sample size needed for 80% power
    """
    df = pd.DataFrame({
        'value': input_data.value_column,
        'time': input_data.time_column,
        'subject': input_data.subject_column
    })

    aov = pg.rm_anova(
        data=df,
        dv='value',
        within='time',
        subject='subject',
        effsize='n2'
    )
    num_conditions = df['time'].nunique()
    num_subjects = df['subject'].nunique()
    power = pg.power_rm_anova(
        eta_squared=aov['n2'][0],
        m=num_conditions,
        n=num_subjects,
        alpha=0.05,
        epsilon=aov['eps'][0]
    )
    n_at_80_power = pg.power_rm_anova(
        eta_squared=aov['n2'][0],
        m=num_conditions,
        power=0.8,
        alpha=0.2,
        epsilon=aov['eps'][0]
    )
    output = {
        'power': power,
        'n at 80% power': n_at_80_power
    }
    return json.dumps(output, allow_nan=True, ensure_ascii=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)