from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pingouin as pg
import json
from typing import List, Optional

app = FastAPI()

class RepeatedMeasuresAnovaInput(BaseModel):
    value_column: List[int]
    group_column: Optional[List[object]]=None
    time_column: List[object]
    subject_column: List[int]


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
        return json.dumps(output, allow_nan=True)
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
    
    return json.dumps(result, allow_nan=True)

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

    return json.dumps(output, allow_nan=True)

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
    return json.dumps(output, allow_nan=True)
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)