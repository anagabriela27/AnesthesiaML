"""
Scope: EDA
Brief: This file contains utility functions for exploratory data analysis (EDA) in Python.

Note: Although the functions in this file work, they may benefit from further optimization and refactoring.
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go


def get_recorded_vital_signs_df(vital_signs_df, vital_signs):
    """
    Generate a DataFrame of recorded vital signs for each patient.

    Args:
        patient_ids (list): List of unique patient IDs.
        vital_signs_df (pandas.DataFrame): DataFrame containing vital signs data.
        vital_signs (list): List of vital signs to consider.

    Returns:
        pandas.DataFrame: DataFrame containing recorded vital signs for each patient.
    """

    # Dictionary to store recorded signs for each patient using defaultdict
    recorded_vitalsigns = defaultdict(list)
    patient_ids = vital_signs_df['caseid'].unique()

    # Loop through patient ids and vital signs
    for patient_id in patient_ids:
        vital_signs_id = vital_signs_df[vital_signs_df['caseid'] == patient_id]

        for sign in vital_signs:
            # Check if there are values different from 0 and null
            if vital_signs_id[sign].dropna().ne(0).any():

                recorded_vitalsigns[patient_id].append(sign)
    # Create DataFrame from defaultdict
    recorded_vital_signs_df = pd.DataFrame({
        'caseid': list(recorded_vitalsigns.keys()),
        'RecordedVitalSigns': [signs for signs in recorded_vitalsigns.values()],
    })

    # Add a column with the number of recorded signs for each patient
    recorded_vital_signs_df['TotalRecordedVitalSigns'] = recorded_vital_signs_df['RecordedVitalSigns'].apply(len)
    recorded_vital_signs_df = recorded_vital_signs_df.sort_values(by='TotalRecordedVitalSigns', ascending=False)
    return recorded_vital_signs_df

def plot_recorded_vital_signs_distribution(recorded_vitalsigns_vital_signs_df):
    """
    Plot the distribution of recorded vital signs for patients.

    Args:
        recorded_vitalsigns_vital_signs_df (pandas.DataFrame): DataFrame containing recorded vital signs data.

    Returns:
        None
    """
    
    # Create a DataFrame to count unique sets of recorded vital signs and their occurrence
    unique_sets_count_vital_signs_df = pd.DataFrame(recorded_vitalsigns_vital_signs_df['RecordedVitalSigns'].apply(frozenset).value_counts())

     # Calculate the number of vital signs for each set and add it as a new column
    unique_sets_count_vital_signs_df['Number of Vital Signs'] = unique_sets_count_vital_signs_df.index.map(len)

    # Set the index column as a regular column and create a new column with recorded vital signs as strings
    unique_sets_count_vital_signs_df.reset_index(inplace=True)
    unique_sets_count_vital_signs_df['RecordedVitalSignsStrings'] = unique_sets_count_vital_signs_df['RecordedVitalSigns'].apply(lambda x: ', '.join(x))

    #Set each list of values in unique_sets_count_vital_signs_df['RecordedVitalSignsStrings'] in alphabetical order
    unique_sets_count_vital_signs_df['RecordedVitalSignsStrings'] = unique_sets_count_vital_signs_df['RecordedVitalSignsStrings'].apply(lambda x: ', '.join(sorted(x.split(', '))))

    #Order unique_sets_count_vital_signs_df['RecordedVitalSignsStrings2'] in alphabetical order
    unique_sets_count_vital_signs_df = unique_sets_count_vital_signs_df.sort_values(by='RecordedVitalSignsStrings')

    # Define custom color scale
    color_scale = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan','mediumvioletred']

    # Sort DataFrame by 'Number of Vital Signs' in ascending order
    unique_sets_count_vital_signs_df = unique_sets_count_vital_signs_df.sort_values(by='Number of Vital Signs',ascending=False)

    # Convert 'Number of Vital Signs' column to string
    unique_sets_count_vital_signs_df['Number of Vital Signs'] = unique_sets_count_vital_signs_df['Number of Vital Signs'].astype(str)

    # Plot the distribution of the RecordedVitalSigns with Plotly
    fig = px.bar(unique_sets_count_vital_signs_df, x='RecordedVitalSignsStrings', y='count', color='Number of Vital Signs',
                title='<b>Distribution of Recorded Vital Signs</b>', labels={'RecordedVitalSignsStrings': '<b> Recorded Vital Signs </b>', 'count': '<b> Number of Patients (log scale) </b>'},
                width=800, height=600, color_discrete_sequence=color_scale, text_auto=True)

    #Update axis
    fig.update_xaxes(tickangle=-70, tickfont=dict(size=12))  # Customize x-axis tick font size and rotation
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')  #Set grid color to light gray

    #Set yaxis in log scale
    fig.update_yaxes(type="log")

    # Set figure title to the middle
    fig.update_layout(title=dict(x=0.5))

    #Set background color to white
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.update_traces(textfont_size=9, textangle=0, textposition="outside", cliponaxis=False,marker_line_color='black', marker_line_width=1)

    # Show the plot
    return fig

def plot_intraop_drugs(clinical_info_df):
    """
    Plot the distribution of intraoperative drugs.

    Args:
        clinical_info_df (pandas.DataFrame): DataFrame containing clinical information data.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot.
    """
    #Information on drugs administered besides Sevoflurane and Desflurane
    intraop_columns = clinical_info_df.columns[clinical_info_df.columns.str.startswith('intraop')][4:]

    #get position of intraop columns
    intraop_columns_pos = [clinical_info_df.columns.get_loc(col) for col in intraop_columns]

    #Replace name of columns to make them more readable
    df_cases_renamed = clinical_info_df.rename(columns={'intraop_crystalloid': 'Crystalloid', 'intraop_colloid': 'Colloid', 
                                                        'intraop_ppf': 'Propofol bolus','intraop_ftn': 'Fentanyl',
                                                          'intraop_mdz': 'Midazolam','intraop_rocu': 'Rorocuronium',
                                                            'intraop_vecu':'Vecuronium','intraop_eph':'Ephedrine',
                                                            'intraop_phe':'Phenylephrine','intraop_epi':'Epinephrine',
                                                            'intraop_ca':'Calcium chloride'})

    #Get the columns in intraop_columns_pos
    intraop_columns = df_cases_renamed.columns[intraop_columns_pos]

    #Count, for each of the intraoperative columns, the number of patients with non-zero values
    nonzero_counts = df_cases_renamed[intraop_columns].ne(0).sum()

    #Plot the number of patients with non-zero values for each of the intraoperative columns 
    fig = plt.figure(figsize=(10,6))

    nonzero_counts = nonzero_counts.sort_values(ascending=False)
    sns.barplot(x=nonzero_counts.values, y=nonzero_counts.index, color='skyblue',edgecolor='black')

    # Add the count values on top of the bars
    for i in range(len(nonzero_counts)):
        plt.text(nonzero_counts.values[i], i, nonzero_counts.values[i], ha = 'left', va = 'center', color='black')
    plt.xlabel('Number of patients')
    plt.ylabel('Intraoperative drugs')
    #plt.title('Number of patients with non-zero values for each of the intraoperative drugs')
    plt.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    return fig

def plot_durations_hists(clinical_info_df):
    """
    Plot histograms of the durations of anesthesia and surgery.

    Args:
        clinical_info_df (pandas.DataFrame): DataFrame containing clinical information data.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot.
    """
    #Create histogram with both anesthesia and surgery durations
    fig = plt.figure(figsize=(10,6))
    sns.histplot(data=clinical_info_df, x='anest_duration_minutes',color='skyblue', bins =20,label='Anesthesia duration')
    sns.histplot(data=clinical_info_df, x='op_duration_minutes', color='#ffb3ba',bins = 20, label='Surgery duration')
    plt.xlabel('Duration (min)')
    plt.ylabel('Number of patients')
    plt.legend()
    plt.grid(True, which='major',axis = 'x', linestyle='--', linewidth=0.5, color='gray',alpha=0.5)

    # Show the plot
    plt.show()  

    plt.tight_layout()
    return fig

def descriptive_histogram(clinical_info_df,column,fig_title, xaxis_label = None,ylabel = 'Number of surgeries',
                          width=10,height=7,dpi = 100, numbers = True,grid = False,nbins=20,nxticks=10,kde=False):
    """
    Plot histograms of the columns of a dataframe and show the number of nulls and percentage of nulls.

    Args:
        clinical_info_df (pandas.DataFrame): DataFrame containing clinical information data.
        column (str): Column name to plot.
        fig_title (str): Title of the figure.
        xaxis_label (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Number of surgeries'.
        width (int, optional): Width of the figure. Defaults to 10.
        height (int, optional): Height of the figure. Defaults to 7.
        dpi (int, optional): Dots per inch for the figure. Defaults to 100.
        numbers (bool, optional): Whether to show numbers on top of each bin. Defaults to True.
        grid (bool, optional): Whether to show grid. Defaults to False.
        nbins (int, optional): Number of bins for the histogram. Defaults to 20.
        nxticks (int, optional): Number of ticks on the x-axis. Defaults to 10.
        kde (bool, optional): Whether to show kernel density estimate. Defaults to False.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='seaborn')

    #Get number of nulls and percentage of nulls in each column
    nulls = clinical_info_df[column].isnull().sum()
    total = clinical_info_df[column].shape[0]
    percentage = (nulls/total)*100


    #Add nulls and percentage of nulls as columns to the description dataframe
    clinical_info_df_description = pd.DataFrame(clinical_info_df[column].describe()).T
    clinical_info_df_description['#Nulls'] = nulls
    clinical_info_df_description['%Nulls'] = round(percentage,3)

    # Create the figure with specified width and height
    fig, ax = plt.subplots(figsize=(width, height),dpi = dpi)

    if column == 'asa':
        clinical_info_df['asa'] = clinical_info_df['asa'].astype(str)

        #Replace 1.0 with I and so on
        # We consider only until ASA = III since we are looking at pre-selected data of patients with ASA < IV
        clinical_info_df['asa'] = clinical_info_df['asa'].replace({'1.0':'I','2.0':'II','3.0':'III','nan':'N/D'})
        order = ['I','II','III','N/D']

        sns.countplot(x='asa', data=clinical_info_df, color='skyblue',edgecolor='black',order=order)

        # Set the x-ticks to be the unique values in the column
        for rect in ax.patches:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.03, int(height), ha='center', va='bottom')
            
        plt.xlabel('ASA Index')
        plt.ylabel('Number of patients')

    else:
        # Plot histogram
        sns.histplot(clinical_info_df.loc[:,column], bins=nbins, color='skyblue', edgecolor='black', ax=ax,kde=kde)
        # Set number of xticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))

        if kde:
            ax.lines[0].set_color('crimson')

    #If we want to add the counts on top of each bin
    if numbers:
        # Add counts on top of each bin
        for rect in ax.patches:
            height = rect.get_height()
            
            if height > 0:
                ax.text(rect.get_x() + rect.get_width() / 2., height + 0.03, int(height), ha='center', va='bottom')

    # #If grid is true, show only y grid, if grid is false, don't show grid
    if grid:
        ax.grid(axis='y',alpha=0.5)
    else:
        ax.grid(False)

    plt.title(fig_title)
    if xaxis_label == None:
        plt.xlabel(column)
    else:
        plt.xlabel(xaxis_label)

    plt.ylabel(ylabel)

    return clinical_info_df_description,fig

def vital_signs_boxplot(vital_signs_df, vital_signs,units, colors=None,size_legend=10,xlegend = 0.7,ylegend = 0.95,dpi=100,image_path=None):
    """
    Description:
    Generates a boxplot for each specified vital sign in the provided DataFrame. 
    Visualizes the distribution of each vital sign's values and includes statistical information such as minimum, maximum, quartiles, and fences.
    Identifies the patients that are outliers for each vital sign.

    Args:
    - vital_signs_df: pandas DataFrame
      Input DataFrame containing the data to be visualized.

    - vital_signs: list
      A list of vital signs for which boxplots are to be generated.

    - units: str
        An ordered list with the unit of measurement for the vital signs.

    - colors: list
      A list of colors to be used for each boxplot.

    - size_legend: int, optional (default=10)
      Font size for the statistical information displayed on the plot.

    - xlegend: float, optional (default=0.7)
      X-coordinate position for the statistical information text box.

    - ylegend: float, optional (default=0.95)
      Y-coordinate position for the statistical information text box.

    - dpi: int, optional (default=100)
      Dots per inch for the figure's resolution.

    Returns:
    - fig: matplotlib Figure object. The generated matplotlib Figure object containing the boxplots for the specified vital signs.
    - outliers_dict: dict. Dictionary containing the patient ids that are outliers for each vital sign.

    """

    if colors is None:
        colors = ['skyblue','mediumvioletred','gold',
          'orange','red','limegreen','navy','lightsteelblue',
          'darkred','beige','darkgreen','darkblue','darkorange']

    for i in range(len(vital_signs)):
        sign = vital_signs[i]
        sign_vital_signs_df = vital_signs_df[['caseid',sign]]

        fig = plt.figure(dpi=dpi)

        # Calculate statistical values
        sign_description = sign_vital_signs_df[sign].describe()

        #Round columns of sign_description to 4 decimal places
        sign_description = sign_description.round(4)

        sns.boxplot(y=sign_vital_signs_df[sign], color=colors[i],orient='v')
        # Calculate lower and upper fences (assuming '25%' and '75%' keys are present in sign_description)
        Q1 = sign_description['25%']
        Q3 = sign_description['75%']
        IQR = Q3 - Q1
        lower_fence = round((Q1 - 1.5 * IQR),4)
        upper_fence = round((Q3 + 1.5 * IQR),4)

        # Create text box with statistical information including lower and upper fences
        plt.text(xlegend, ylegend,
                f"Min: {sign_description['min']}\n"
                f"Q1: {Q1}\n"
                f"Median: {sign_description['50%']}\n"
                f"Q3: {Q3}\n"
                f"Max: {sign_description['max']}\n"
                f"Lower Fence: {lower_fence}\n"
                f"Upper Fence: {upper_fence}",
                fontsize=size_legend,
                verticalalignment='top', horizontalalignment='left',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='lightgray', alpha=0.5))
        

        #Set y grid
        plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

        if sign not in ['BIS']:
            plt.ylabel(f'Values ({units[sign]})')

        else:
            #BIS is unitless
            plt.ylabel('Values')
        
        if sign == 'SPO2':
            #Set yaxis to logarithmic scale
            plt.yscale('log')

            #Set yaxis label
            plt.ylabel('Values (%,logscale)')

        #Put the title a little bit higher
        plt.title(f'{sign} values Boxplot', y=1.02)

        #Save in svg format
        if image_path is not None:
            plt.savefig(image_path+sign+'.svg', format='svg')
    return fig
    
def get_timediffs(vital_signs_df, vital_signs,rows_timediff=1):
    """
    Description:
    This function calculates the time differences between consecutive non-null values of vital signs in a DataFrame grouped by 'caseid'. 
    It iterates over each specified vital sign and computes the time differences for each group separately.

    Args:
    - vital_signs_df: pandas DataFrame
      Input DataFrame containing data to be processed.

    - vital_signs: list
      A list of vital signs for which time differences are to be calculated (are columns in the DataFrame)

    - rows_timediff: int, default=1
      The time difference between consecutive rows in the DataFrame. The default value is 1, which corresponds to 1 minute.

    Returns:
    - vital_signs_df_sr: pandas DataFrame
      A copy of the input DataFrame with additional columns containing time differences for each vital sign specified in 'vital_signs'.
    The time differences are computed in minutes and added as new columns with names formatted as 'diffs_vital_sign'.
    """
    import pandas as pd

    vital_signs_df_sr = vital_signs_df.copy()

    # Iterate over each vital sign
    for vital_sign in vital_signs:
        # Group by 'caseid'
        grouped = vital_signs_df_sr.groupby('caseid')
        
        # Initialize an empty list to store time differences
        time_diffs = []
        
        # Iterate over each group
        for name, group in grouped:
            # Get the indices of non-null values
            non_null_indices = group[vital_sign].notnull()

            # Calculate the time differences between non-null values based on their index positions
            non_null_index_positions = group.loc[non_null_indices].index
            diffs = pd.Series(non_null_index_positions).diff().fillna(0) * rows_timediff  # converting the difference to time in seconds
            #diffs = diffs / 60.0  # converting seconds to minutes
            diffs.index = non_null_index_positions
            diffs.name = f'diffs_{vital_sign}'

            # Join the time differences back to the original group
            group = group.join(diffs, how='outer')
            
            # Append the calculated time differences to the list
            time_diffs.append(group)
        
        # Concatenate the groups back into a DataFrame
        vital_signs_df_sr = pd.concat(time_diffs)
        
    # Dropping the original vital sign columns if needed
    vital_signs_df_sr = vital_signs_df_sr.drop(columns=vital_signs)

    return vital_signs_df_sr

def plot_timeseries(vital_signs_df,id,vital_signs,secondary_axis_signs=None,xaxis_range=False,height=800,width=1300,startend=None, markers=True, connectgaps=True,colors = None):
    """
        Description:
        This function generates a plotly figure with the time series of the specified vital signs for a given patient ID.

        Args:
        - vital_signs_df: pandas DataFrame.DataFrame containing the vital signs data.
        - id: int. The patient ID.
        - vital_signs: list. A list of vital signs to be included in the plot.
        - xaxis_range: bool, optional (default=False). If True, set the x-axis range to the minimum and maximum values of the 'Data' column.
        - height: int, optional (default=800). The height of the plot in pixels.
        - width: int, optional (default=1300). The width of the plot in pixels.
        - startend: list, optional (default=None). A list containing the start and end dates for the x-axis range.
        - markers: bool, optional (default=True). If True, display markers on the plot.
        - connectgaps: bool, optional (default=True). If True, connect the gaps in the data.

        Returns:
        - fig: plotly.graph_objs._figure.Figure. The generated plotly figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    vital_signs_id = vital_signs_df[vital_signs_df['caseid']==id]

    if colors == None:
        colors = ['purple','orange','green','blue','red','orange']
    signs_legend_ymain = []
    signs_legend_ysec = []

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if markers:
        mode = 'lines+markers'

    else:
        mode = 'lines'

    if secondary_axis_signs is None:
        secondary_axis_signs = []

    for sign in vital_signs:
        #IF the sign in inDESS or inSEV, add it to the secondary axis
        if sign in secondary_axis_signs:
            fig.add_trace(go.Scatter(x=np.array(vital_signs_id[['time', sign]]['time']), 
                                y=vital_signs_id[sign], 
                                name=sign,
                                mode=mode,  # Add markers
                                line=dict(color=colors[vital_signs.index(sign)]),
                                marker=dict(size =4)), 
                                secondary_y=True)

            signs_legend_ysec = signs_legend_ysec + [sign]        
            
        else:    
            fig.add_trace(go.Scatter(x=np.array(vital_signs_id[['time', sign]]['time']), 
                                y=vital_signs_id[sign], 
                                name=sign,
                                mode=mode,  # Add markers
                                line=dict(color=colors[vital_signs.index(sign)]),
                                marker=dict(size =4)), 
                                secondary_y=False)
            signs_legend_ymain = signs_legend_ymain + [sign]
        
                
        fig.update_traces(connectgaps=connectgaps)

        #Set x axis limits
        if xaxis_range:
            data_min = vital_signs_id.loc[vital_signs_id[vital_signs].dropna(how='all').index[0]]['time'] #Get the first data point with non-null values in the vital signs
            data_max = vital_signs_id.loc[vital_signs_id[vital_signs].dropna(how='all').index[-1]]['time'] #Get the last data point with non-null values in the vital signs
            fig.update_xaxes(range=[data_min, data_max])

        #Set y axis title
        fig.update_yaxes(title_text=f"{signs_legend_ymain} Values", secondary_y=False)
        fig.update_yaxes(title_text=f"{signs_legend_ysec} Values", secondary_y=True)

        fig.update_layout(title_text=f"Vital signs: {vital_signs} - Patient {id}", xaxis_title='<b> Time </b>',
                            title=dict(x=0.5, xanchor='center', yanchor='top'), title_font_size=18,height=height,width=width)
        
    return fig

def plot_multiple_timeseries(vital_signs_df,id,vital_signs,
                   markers=True,height=800, width=1400,xlegend=None,ylegend=None,size_legend=12,connectgaps=True, mode='lines',):
    """
    Function to plot multiple time series for a given patient id

        Args:
        - vital_signs_df: pandas DataFrame.DataFrame containing the vital signs data.
        - id: int. The patient ID.
        - vital_signs: list. A list of vital signs to be included in the plot.
        - markers: bool, optional (default=True). If True, display markers on the plot.
        - xaxis_range: str, optional (default='nonnulldata'). The range of the x-axis. Possible values are 'nonnulldata', 'startend', or None.
        - height: int, optional (default=800). The height of the plot in pixels.
        - width: int, optional (default=1400). The width of the plot in pixels.
        - xlegend: float, optional (default=None). The x-coordinate position for the legend.
        - ylegend: float, optional (default=None). The y-coordinate position for the legend.
        - size_legend: int, optional (default=12). The font size for the legend.

        Returns:
        - fig: plotly.graph_objs._figure.Figure. The generated plotly figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np

    if markers: 
        mode = 'lines+markers'
    elif markers == False:
        mode = 'lines'        

    vital_signs_id = vital_signs_df[vital_signs_df['caseid']==id]
    
    #Convert to minutes
    vital_signs_id['time'] = vital_signs_id['time'].apply(lambda x: x/60)


    colors = ['purple','orange','green','maroon','blue','crimson','yellowgreen','seagreen','red','lightblue']

    nodatasigns = []

    for signal in vital_signs:
        sign_data = vital_signs_id[signal]

        #If all values are either null or 0, then there is no data for this signal
        if (sign_data[sign_data.notna()].eq(0).all()) or (sign_data.isna().all()):
            print(f'No data for {signal}')
            nodatasigns.append(signal)  # Add sign to the removal list

    # Remove signals with no data from the list of vital_signs to plot
    for signal in nodatasigns:
        vital_signs.remove(signal)

    # Create a new figure with subplots for each vital sign
    num_signals = len(vital_signs)
    num_cols = int(np.ceil(np.sqrt(num_signals)))
    num_rows = int(np.ceil(num_signals / num_cols))

    fig = make_subplots(rows=num_rows, cols=num_cols)  

    # Iterate over each sign
    for i, sign in enumerate(vital_signs, start=1):
        sign_data = vital_signs_id[['time',sign]]

        row = (i - 1) // num_cols + 1
        col = (i - 1) % num_cols + 1

        # Add trace each sign in a different subplot
        if sign != 'MBP' and sign != 'SBP' and sign != 'DBP':
            # Add data for the current sign
            fig.add_trace(go.Scatter(x=sign_data['time'], y=sign_data[sign],
                                    mode=mode,name=sign,
                                    line=dict(color=colors[vital_signs.index(sign)],width = 1),
                                    marker=dict(color=colors[vital_signs.index(sign)],size = 3)),row=row, col=col)
            
            fig.update_yaxes(title_text=f"<b> {sign} Values</b>",
                            tickfont=dict(size=10), gridcolor='white', title_font=dict(size=12),row=row, col=col,title=dict(standoff=0))
            
            fig.update_xaxes(title_text= "<b> Time  </b>",gridcolor='white', matches='x',row=row, col=col)
        
            
        ############################## Join SBP, MBP and DBP in the same subplot ###################################
        else: #When sign is MBP, add traces for 'DBP' and 'SBP' with different colors
            signs_T = ['SBP','MBP','DBP']
        
            # Select specific colors from the 'Viridis' color scale for this subplot
            t_colors =[px.colors.sequential.Viridis[3],px.colors.sequential.Viridis[0],px.colors.sequential.Viridis[5]]
            fig.update_yaxes(title_text="<b> Blood Pressure Values (mmHg)</b>",
                                tickfont=dict(size=10), gridcolor='#f7f7f5', title_font=dict(size=10),row=row, col=col,title=dict(standoff=0))
            
            for sign_T, color_T in zip(signs_T, t_colors):
                sign_T_data = vital_signs_id[['time',sign_T]]
                
                fig.add_trace(go.Scatter(x=sign_T_data['time'], y=sign_T_data[sign_T],
                                            mode=mode, name=sign_T,
                                            line=dict(color=color_T,width = 1), marker=dict(color=color_T,size = 3)), row=row, col=col)
            
                fig.update_xaxes(title_text= "<b> Time  </b>",gridcolor='white', matches='x',row=row, col=col)


    # Update layout to place 
    #Set title in the middle of the figure
    fig.update_layout(legend=dict(
            # Adjust the y position of the legend
                bgcolor='#f7f7f5',
                bordercolor='gray',
                borderwidth=2,
                font = dict(size=12)
            ))
    fig.update_layout(height=height, width=width, title_text=f"<b> Vital Signs Time Series for Patient {id} </b>",
                    title=dict(x=0.5, xanchor='center', yanchor='top'),plot_bgcolor='#f7f7f5')

    if (xlegend != None) and (ylegend != None):
        xlegend = xlegend
        ylegend = ylegend

        fig.update_layout(legend=dict(
            x=xlegend,
            y=ylegend,
            font = dict(size=size_legend)
        ))

    fig.update_traces(connectgaps=connectgaps)

    return fig 

def plot_multiple_timeseries_mplb(vital_signs_df, id, vital_signs,
                                  general_info_df=None, startendline=False,
                                width=12, height=8, xlegend=1.25, ylegend=0.8, size_legend=10, title=True):
    """
    Plots the time series of the vital signs for a given patient.

    Args:
    vital_signs_df (pd.DataFrame): DataFrame containing the vital signs data.
    id (int): ID of the patient to plot.
    vital_signs (list): List of vital signs to plot.
    general_info_df (pd.DataFrame): DataFrame containing the general information of the patients. Default is None.
    startendline (bool): Whether to plot vertical lines for the start and end of the surgery. Default is False.
    width (int): Width of the figure. Default is 12.
    height (int): Height of the figure. Default is 8.
    xlegend (float): x-coordinate of the legend. Default is 1.25.
    ylegend (float): y-coordinate of the legend. Default is 0.8.
    size_legend (int): Size of the legend. Default is 10.
    title (bool): Whether to add a title to the plot. Default is True.

    Returns:
    fig (matplotlib.figure.Figure): Figure containing the plot.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    
    vital_signs_id = vital_signs_df[vital_signs_df['caseid'] == id]


    #Conver to minutes
    vital_signs_id['time'] = vital_signs_id['time'].apply(lambda x: x/60)

    colors = ['purple', 'orange', 'green', 'darksalmon', 'navy', 'crimson', 'yellowgreen', 'seagreen', 'red']

    nodatasigns = []

    for signal in vital_signs:
        sign_data = vital_signs_id[signal]

        # If all values are either null or 0, then there is no data for this signal
        if (sign_data[sign_data.notna()].eq(0).all()) or (sign_data.isna().all()):
            print(f'No data for {signal}')
            nodatasigns.append(signal)  # Add sign to the removal list

    # Remove signals with no data from the list of vital_signs to plot
    for signal in nodatasigns:
        vital_signs.remove(signal)

    # Create a new figure with subplots for each vital sign
    num_signals = len(vital_signs)
    num_cols = int(np.ceil(np.sqrt(num_signals)))
    num_rows = int(np.ceil(num_signals / num_cols))

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(width, height))  # Adjust figure size

    # Iterate over each sign
    for i, sign in enumerate(vital_signs, start=1):
        sign_data = vital_signs_id[['time', sign]].dropna(subset=[sign])


        row = (i - 1) // num_cols
        col = (i - 1) % num_cols

        ax = axes[row, col] if num_signals > 1 else axes

        if sign not in ['MBP', 'SBP', 'DBP']:
            # Plot the current signal
            ax.plot(sign_data['time'], sign_data[sign], linestyle='-', color=colors[vital_signs.index(sign)], label=sign)
            ax.set_title(f"{sign} ")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Values", labelpad=5)

        # Join all blood pressures in the same subplot
        else:  # Only need to do this for MBP as it's the main BP plot
            ax.plot(vital_signs_id.dropna(subset=['SBP'])['time'], vital_signs_id.dropna(subset=['SBP'])['SBP'],
                    linestyle='-', color='teal', label='SBP')
            ax.plot(vital_signs_id.dropna(subset=['MBP'])['time'], vital_signs_id.dropna(subset=['MBP'])['MBP'],
                    linestyle='dashdot', color='slateblue', label='MBP')
            ax.plot(vital_signs_id.dropna(subset=['DBP'])['time'], vital_signs_id.dropna(subset=['DBP'])['DBP'],
                    linestyle='--', color='lightseagreen', label='DBP')

            ax.set_title("Blood Pressure")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Values", labelpad=5)

        # Add lines with the start and finish surgery times (if applicable)
        if startendline and general_info_df is not None:
            general_info_id = general_info_df[general_info_df['caseid'] == id]
            start_surgery = general_info_id['opstart'].values[0]/60
            end_surgery = general_info_id['opend'].values[0]/60
            start_anesthesia = general_info_id['anestart'].values[0]/60
            end_anesthesia = general_info_id['aneend'].values[0]/60
            ax.axvline(start_surgery, color='darkred', linestyle='--', label='Start/End Surgery' if i == len(vital_signs) else None)
            ax.axvline(end_surgery, color='darkred', linestyle='--')
            ax.axvline(start_anesthesia, color='crimson', linestyle='--', label='Start/End Anesthesia' if i == len(vital_signs) else None)
            ax.axvline(end_anesthesia, color='crimson', linestyle='--')

    # Hide any unused subplots
    if num_signals < num_rows * num_cols:
        for j in range(num_signals, num_rows * num_cols):
            fig.delaxes(axes.flatten()[j])
            
    # Iterate over each axis and enable horizontal grid lines
    for ax in axes.flat:
        ax.grid(axis='y', linestyle='--', color='gray', linewidth=0.5, alpha=0.3)

    # Add legend for the entire plot and adjust its position
    if xlegend is not None and ylegend is not None:
        fig.legend(bbox_to_anchor=(xlegend, ylegend), fontsize=size_legend)
    else:
        fig.legend(fontsize=size_legend, loc='best')

    if title:
        plt.suptitle(f"Vital Signs Time Series for Patient {id}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    return plt.gca().figure

def plot_correlation_heatmap(vital_signs_df_vs_cleaned, tickssize=16, width=800, height=700):
    """
        Plots a heatmap of the correlation matrix of the Z-score normalized data using Plotly.
        
        Arguments:
            vital_signs_df_vs_cleaned (pd.DataFrame): The DataFrame containing the vital signs data.
            tickssize (int): Font size for the tick labels.
            width (int): Width of the plot.
            height (int) Height of the plot. 

        Returns:
            fig : plotly.graph_objects.Figure. The Plotly figure object containing the heatmap.
    """

    # Group the dataframe by 'caseid'
    grouped_vital_signs_df = vital_signs_df_vs_cleaned.groupby('caseid')

    # Define a function to Z-score normalize the data for each column within each group
    def zscore_normalize(group):
        return group.apply(lambda x: (x - x.mean()) / x.std() if x.name not in ['caseid', 'time'] else x)

    # Apply Z-score normalization
    zscore_normalized_df = grouped_vital_signs_df.apply(zscore_normalize).reset_index(drop=True)

    # Compute the correlation matrix
    correlation_matrix = zscore_normalized_df.drop(columns=['caseid', 'time']).corr()

    # Mask upper triangle
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    correlation_matrix_masked = correlation_matrix.mask(mask)
    np.fill_diagonal(correlation_matrix_masked.values, 1)

    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix_masked.values,
        x=correlation_matrix_masked.columns,
        y=correlation_matrix_masked.index,
        zmid=0,
        colorscale='RdBu_r',
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        title='Correlation Heatmap (Z-score normalized)',
        width=width,
        height=height,
        xaxis=dict(tickfont=dict(size=tickssize)),
        yaxis=dict(tickfont=dict(size=tickssize)),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig.show()
