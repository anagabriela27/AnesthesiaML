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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------------
# Patient Info & Vital Sign Coverage
# -----------------------------------------------------------------------------------

def get_recorded_vital_signs_df(vital_signs_df, vital_signs):
    """
    Generate a DataFrame of recorded vital signs for each patient.

    Arguments:
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

def plot_recorded_vital_signs_distribution(df):
    """
    Plot the distribution of recorded vital signs for patients.

    Parameters:
        df (pandas.DataFrame): DataFrame with a 'RecordedVitalSigns' column.

    Returns:
        plotly.graph_objects.Figure: Bar plot figure.
    """
    # Count unique combinations of recorded vital signs
    vital_sign_sets = df['RecordedVitalSigns'].apply(frozenset).value_counts().reset_index()
    vital_sign_sets.columns = ['RecordedVitalSigns', 'count']
    
    # Add a column with the number of recorded signs for each combination
    vital_sign_sets['Number of Vital Signs'] = vital_sign_sets['RecordedVitalSigns'].map(len)
    vital_sign_sets['RecordedVitalSignsStrings'] = vital_sign_sets['RecordedVitalSigns'].apply(lambda x: ', '.join(sorted(x)))

    # Order the DataFrame by the number of vital signs and then by the string representation
    vital_sign_sets = vital_sign_sets.sort_values(by=['Number of Vital Signs', 'RecordedVitalSignsStrings'], ascending=[False, True])
    vital_sign_sets['Number of Vital Signs'] = vital_sign_sets['Number of Vital Signs'].astype(str)

    # Create a color scale for the bars
    color_scale = ['purple', 'lightseagreen','skyblue', 'green', 'red', 'yellow', 'cyan','mediumvioletred']

    # Plot the distribution of recorded vital signs
    fig = px.bar(
        vital_sign_sets, x='RecordedVitalSignsStrings', y='count', color='Number of Vital Signs',
        title='<b>Distribution of Recorded Vital Signs</b>',
        labels={
            'RecordedVitalSignsStrings': '<b> Recorded Vital Signs </b>',
            'count': '<b> Number of Patients (log scale) </b>'
        },
        width=1000, height=800, color_discrete_sequence=color_scale, text_auto=True
    )

    fig.update_layout(
        title=dict(x=0.5),
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    fig.update_xaxes(tickangle=-70, tickfont=dict(size=12))
    fig.update_yaxes(type="log", showgrid=False)
    fig.update_traces(
        textfont_size=9, textangle=0, textposition="outside",
        cliponaxis=False, marker_line_color='black', marker_line_width=1
    )

    return fig

# -----------------------------------------------------------------------------------
# Clinical Data Distributions
# -----------------------------------------------------------------------------------

def plot_intraop_drugs(clinical_info_df):
    """
    Plot the distribution of intraoperative drugs.

    Arguments:
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
    counts = df_cases_renamed[intraop_columns].ne(0).sum().sort_values(ascending=False)

    #Plot the number of patients with non-zero values for each of the intraoperative columns 
    fig = plt.figure(figsize=(10,6))

    sns.barplot(x=counts.values, y=counts.index, color='skyblue',edgecolor='black')

    # Add the count values on top of the bars
    for i, val in enumerate(counts.values):
        plt.text(val, i, val, ha='left', va='center')
    plt.xlabel('Number of patients')
    plt.ylabel('Intraoperative drugs')
    plt.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    return fig

def plot_durations_hists(clinical_info_df):
    """
    Plot histograms of the durations of anesthesia and surgery.

    Arguments:
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

    Arguments:
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


# -----------------------------------------------------------------------------------
# Vital Signs Analysis
# -----------------------------------------------------------------------------------

def vital_signs_boxplot(vital_signs_df, vital_signs, units, colors=None,
                        size_legend=10, xlegend=0.7, ylegend=0.95, dpi=100, image_path=None):
    """
        Generates a boxplot for each specified vital sign in the provided DataFrame.
        Arguments:
            vital_signs_df (pandas.DataFrame): DataFrame containing vital signs data.
            vital_signs (list): List of vital signs to plot.
            units (dict): Dictionary mapping vital signs to their units.
            colors (list, optional): List of colors for the boxplots. Defaults to None.
            size_legend (int, optional): Font size for the legend text. Defaults to 10.
            xlegend (float, optional): X-coordinate for the legend text. Defaults to 0.7.
            ylegend (float, optional): Y-coordinate for the legend text. Defaults to 0.95.
            dpi (int, optional): Dots per inch for the figure. Defaults to 100.
            image_path (str, optional): Path to save the images. If None, images are not saved. Defaults to None.
        
        Returns:
            list: List of generated figures. If only one figure is created, it is returned directly.
    """

    if colors is None:
        colors = ['skyblue', 'mediumvioletred', 'gold', 'orange', 'red', 'limegreen',
                  'navy', 'lightsteelblue', 'darkred', 'beige', 'darkgreen', 'darkblue', 'darkorange']

    figs = []

    for sign, color in zip(vital_signs, colors):
        fig = plt.figure(dpi=dpi)
        data = vital_signs_df[sign]

        stats = data.describe().round(4)
        Q1, Q3 = stats['25%'], stats['75%']
        IQR = Q3 - Q1
        lower_fence = round(Q1 - 1.5 * IQR, 4)
        upper_fence = round(Q3 + 1.5 * IQR, 4)

        sns.boxplot(y=data, color=color, orient='v')
        plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

        label = (
            'Values (%,logscale)' if sign == 'spo2'
            else f"Values ({units.get(sign, '')})" if sign != 'bis'
            else 'Values'
        )
        if sign == 'spo2':
            plt.yscale('log')

        plt.ylabel(label)
        plt.title(f'{sign} values Boxplot', y=1.02)

        # Add stats text box
        plt.text(xlegend, ylegend,
                 f"Min: {stats['min']}\nQ1: {Q1}\nMedian: {stats['50%']}\nQ3: {Q3}\n"
                 f"Max: {stats['max']}\nLower Fence: {lower_fence}\nUpper Fence: {upper_fence}",
                 fontsize=size_legend, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='lightgray', alpha=0.5))

        if image_path:
            plt.savefig(f"{image_path}{sign}.svg", format='svg')

        figs.append(fig)

    return figs[0] if len(figs) == 1 else figs
    
def get_timediffs(vital_signs_df, vital_signs, rows_timediff=1):
    """
    Calculates time differences between consecutive non-null values of specified vital signs,
    grouped by 'caseid'.
    
    Arguments:
        vital_signs_df (pandas.DataFrame): DataFrame containing vital signs data.
        vital_signs (list): List of vital signs to calculate time differences for.
        rows_timediff (int): Number of rows to consider for time difference calculation. Default is 1.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for time differences of specified vital signs.
    """
    df = vital_signs_df.reset_index(drop=True).copy()  # evita criar a coluna 'index'

    for sign in vital_signs:
        def compute_diffs(group):
            idx = group[sign].notnull()
            diffs = group.loc[idx].index.to_series().diff().fillna(0) * rows_timediff
            group[f'diffs_{sign}'] = diffs
            return group

        df = df.groupby('caseid', group_keys=False).apply(compute_diffs)

    return df.drop(columns=vital_signs)

def plot_correlation_heatmap(vital_signs_df, tickssize=16, width=800, height=700,
                              fig_title="Correlation Matrix between Vital Signs"):
    """
    Plots a heatmap of the correlation matrix of the Z-score normalized data using Plotly.
    Shows only the lower triangle of the matrix (including the diagonal).

    Arguments:
        vital_signs_df (pandas.DataFrame): DataFrame containing the vital signs data.
        tickssize (int, optional): Font size for the x and y axis ticks. Defaults to 16.
        width (int, optional): Width of the figure. Defaults to 800.
        height (int, optional): Height of the figure. Defaults to 700.
        fig_title (str, optional): Title of the figure. Defaults to "Correlation Matrix between Vital Signs".

    Returns:
        plotly.graph_objects.Figure: The generated heatmap figure.
    """
    # Normalize the data using Z-score
    def zscore_normalize(group):
        return group.apply(lambda x: (x - x.mean()) / x.std() if x.name not in ['caseid', 'time'] else x)

    # Apply Z-score normalization to each caseid and reset the index
    zscore_normalized_df = vital_signs_df.groupby('caseid').apply(zscore_normalize).reset_index(drop=True)

    # Calculate the correlation matrix
    correlation_matrix = zscore_normalized_df.drop(columns=['caseid', 'time']).corr()

    # Boolean mask for the lower triangle of the correlation matrix
    mask = np.tril(np.ones(correlation_matrix.shape), k=0).astype(bool)

    # Replace the upper triangle with None
    correlation_values_masked = correlation_matrix.copy()
    correlation_values_masked[~mask] = None

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_values_masked.values,
        x=correlation_values_masked.columns,
        y=correlation_values_masked.index,
        zmid=0,
        colorscale='RdBu_r',
        colorbar=dict(title='Correlation'),
        zmin=-1, zmax=1
    ))

    # Set the layout of the heatmap
    fig.update_layout(
        title=fig_title,
        title_x=0.5,
        width=width,
        height=height,
        xaxis=dict(tickfont=dict(size=tickssize)),
        yaxis=dict(tickfont=dict(size=tickssize), autorange='reversed'),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

# -----------------------------------------------------------------------------------
# Vital Signs Time Series Plotting
# -----------------------------------------------------------------------------------

def plot_timeseries(vital_signs_df,caseid,vital_signs,general_info_df = None,secondary_axis_signs=None,height=800,width=1300,
                    startendline=True, markers=True, connectgaps=True,color_map = None):
    """
        Generates a plotly figure with the time series of the specified vital signs for a given patient ID.

        Arguments:
        - vital_signs_df: pandas DataFrame.DataFrame containing the vital signs data.
        - caseid: int. The patient ID.
        - vital_signs: list. A list of vital signs to be included in the plot.  
        - general_info_df: pandas DataFrame. DataFrame containing general information about the patient.      
        - height: int, optional (default=800). The height of the plot in pixels.
        - width: int, optional (default=1300). The width of the plot in pixels.
        - startendline: list, optional (default=None). A list containing the start and end dates for the x-axis range.
        - markers: bool, optional (default=True). If True, display markers on the plot.
        - connectgaps: bool, optional (default=True). If True, connect the gaps in the data.

        Returns:
        - fig: plotly.graph_objs._figure.Figure. The generated plotly figure.
    """
    # Color mapping

    if color_map is None:
        color_map = {
            'sbp': 'teal', 'mbp': 'slateblue', 'dbp': 'lightseagreen',
            'hr': 'navy', 'spo2': 'darkorange', 'bis': 'purple',
            'insp_sevo': 'seagreen', 'exp_sevo': 'yellowgreen'
        }
        
    vital_signs_id = vital_signs_df[vital_signs_df['caseid'] == caseid].copy()
    vital_signs_id['time'] = vital_signs_df['time'] / 60  # convert to minutes

    
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
                                line=dict(color=color_map[sign]),
                                marker=dict(size =4)), 
                                secondary_y=True)

            signs_legend_ysec = signs_legend_ysec + [sign]        
            
        else:    
            fig.add_trace(go.Scatter(x=np.array(vital_signs_id[['time', sign]]['time']), 
                                y=vital_signs_id[sign],
                                name=sign,
                                mode=mode,  # Add markers
                                line=dict(color=color_map[sign]),
                                marker=dict(size =4)),
                                secondary_y=False)
            signs_legend_ymain = signs_legend_ymain + [sign]
        
                
        fig.update_traces(connectgaps=connectgaps)

        #Set y axis title
        fig.update_yaxes(title_text=f"{signs_legend_ymain} Values", secondary_y=False)
        fig.update_yaxes(title_text=f"{signs_legend_ysec} Values", secondary_y=True)

        
    # Add vertical lines using general_info_df
    if startendline and general_info_df is not None:
        general_info_case = general_info_df[general_info_df['caseid'] == caseid]
        if not general_info_case.empty:
            opstart, opend = general_info_case.iloc[0][['opstart', 'opend']] / 60
            anestart, aneend = general_info_case.iloc[0][['anestart', 'aneend']] / 60

            if not np.isnan(anestart):
                fig.add_vline(x=anestart, line_dash="dash", line_color="crimson", line_width=2)
            if not np.isnan(aneend):
                fig.add_vline(x=aneend, line_dash="dash", line_color="crimson", line_width=2)
            if not np.isnan(opstart):
                fig.add_vline(x=opstart, line_dash="dash", line_color="darkred", line_width=2)
            if not np.isnan(opend):
                fig.add_vline(x=opend, line_dash="dash", line_color="darkred", line_width=2)

        # Add invisible traces to include dashed lines in the legend
        if not (np.isnan(aneend) and np.isnan(anestart)):
            fig.add_trace(
                go.Scatter(
                    x=[aneend, aneend],
                    y=[0, 1],
                    mode='lines',
                    name="Anesthesia Start/End",
                    line=dict(color="crimson", dash="dash", width=2),  # Use width > 0 for the legend
                    showlegend=True
                )
            )
        if not (np.isnan(opstart) and np.isnan(opend)):
            fig.add_trace(
                go.Scatter(
                    x=[opstart, opstart],
                    y=[0, 1],
                    mode='lines',
                    name="Surgery Start/End",
                    line=dict(color="darkred", dash="dash", width=2),  # Use width > 0 for the legend
                    showlegend=True
                )
            )

        fig.update_layout(title_text=f"Vital signs: {vital_signs} - Patient {caseid}",
                           xaxis_title='<b> Time </b>',
                            title=dict(x=0.5, xanchor='center', yanchor='top'),
                              title_font_size=18,height=height,width=width)
        
    return fig

def plot_multiple_timeseries_plotly(vital_signs_df, general_info_df, caseid, vital_signs,
                                    markers=True, height=800, width=1400,
                                    xlegend=None, ylegend=None,
                                    size_legend=12, connectgaps=True, startendline=True):
    """
    Function to plot multiple time series for a given patient using Plotly,
    with optional vertical lines for surgery and anesthesia events.
    This function does the same as plot_multiple_series_mplb but uses Plotly for plotting.

    Arguments:
        - vital_signs_df (pd.DataFrame): Vital signs time-series data.
        - general_info_df (pd.DataFrame): Info with start/end times for surgery/anesthesia.
        - caseid (int): Patient ID.
        - vital_signs (list): List of vital signs to plot.
        - markers (bool): Show markers in the plot.
        - height, width (int): Plot dimensions.
        - xlegend, ylegend (float): Legend positioning.
        - size_legend (int): Font size of the legend.
        - connectgaps (bool): Connect missing data.
        - startendline (bool): Show vertical lines for surgery/anesthesia events.
    
    Returns:
        - fig (go.Figure): Plotly figure object.
    """
    mode = 'lines+markers' if markers else 'lines'

    df_case = vital_signs_df[vital_signs_df['caseid'] == caseid].copy()
    df_case['time'] = df_case['time'] / 60  # convert to minutes

    # Remove vital signs with no valid values
    valid_signs = []
    for sign in vital_signs:
        values = df_case[sign]
        if not (values.isna().all() or (values[values.notna()] == 0).all()):
            valid_signs.append(sign)
        else:
            print(f"No data for {sign}")

    if not valid_signs:
        raise ValueError("No valid signals to plot.")

    bp_signals = {'sbp', 'mbp', 'dbp'}
    bp_included = any(sign in bp_signals for sign in valid_signs)
    non_bp_signs = [s for s in valid_signs if s not in bp_signals]
    total_plots = len(non_bp_signs) + (1 if bp_included else 0)

    num_cols = int(np.ceil(np.sqrt(total_plots)))
    num_rows = int(np.ceil(total_plots / num_cols))

    fig = make_subplots(rows=num_rows, cols=num_cols)

    # Color palette
    # colors = ['purple', 'orange', 'green', 'maroon', 'blue', 'crimson',
    #           'yellowgreen', 'seagreen', 'red', 'lightblue']
    # color_map = {sign: colors[i % len(colors)] for i, sign in enumerate(valid_signs)}
    color_map = {
            'sbp': 'teal', 'mbp': 'slateblue', 'dbp': 'lightseagreen',
            'hr': 'navy', 'spo2': 'darkorange', 'bis': 'purple',
            'insp_sevo': 'seagreen', 'exp_sevo': 'yellowgreen'
        }

    i = 0  # subplot index
    for sign in non_bp_signs:
        row, col = divmod(i, num_cols)
        row += 1
        col += 1
        fig.add_trace(
            go.Scatter(
                x=df_case['time'],
                y=df_case[sign],
                mode=mode,
                name=sign,
                line=dict(color=color_map[sign], width=1),
                marker=dict(size=3)
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text=f"<b>{sign} Values</b>", row=row, col=col)
        fig.update_xaxes(title_text="<b>Time (min)</b>", matches='x', row=row, col=col)
        i += 1

    if bp_included:
        row, col = divmod(i, num_cols)
        row += 1
        col += 1
        bp_order = ['sbp', 'mbp', 'dbp']
        bp_colors = {'sbp': 'teal', 'mbp': 'slateblue', 'dbp': 'lightseagreen'}

        for bp in bp_order:
            if bp in valid_signs:
                fig.add_trace(
                    go.Scatter(
                        x=df_case['time'],
                        y=df_case[bp],
                        mode=mode,
                        name=bp,
                        line=dict(color=bp_colors[bp], width=1),
                        marker=dict(size=3)
                    ),
                    row=row, col=col
                )
        fig.update_yaxes(title_text="<b>Blood Pressure (mmHg)</b>", row=row, col=col)
        fig.update_xaxes(title_text="<b>Time (min)</b>", matches='x', row=row, col=col)

    # Add vertical lines using general_info_df
    if startendline and general_info_df is not None:
        general_info_case = general_info_df[general_info_df['caseid'] == caseid]
        if not general_info_case.empty:
            opstart, opend = general_info_case.iloc[0][['opstart', 'opend']] / 60
            anestart, aneend = general_info_case.iloc[0][['anestart', 'aneend']] / 60

            if not np.isnan(anestart):
                fig.add_vline(x=anestart, line_dash="dash", line_color="crimson", line_width=2)
            if not np.isnan(aneend):
                fig.add_vline(x=aneend, line_dash="dash", line_color="crimson", line_width=2)
            if not np.isnan(opstart):
                fig.add_vline(x=opstart, line_dash="dash", line_color="darkred", line_width=2)
            if not np.isnan(opend):
                fig.add_vline(x=opend, line_dash="dash", line_color="darkred", line_width=2)

        # Add invisible traces to include dashed lines in the legend
        if not (np.isnan(aneend) and np.isnan(anestart)):
            fig.add_trace(
                go.Scatter(
                    x=[aneend, aneend],
                    y=[0, 1],
                    mode='lines',
                    name="Anesthesia Start/End",
                    line=dict(color="crimson", dash="dash", width=2),  # Use width > 0 for the legend
                    showlegend=True
                )
            )
        if not (np.isnan(opstart) and np.isnan(opend)):
            fig.add_trace(
                go.Scatter(
                    x=[opstart, opstart],
                    y=[0, 1],
                    mode='lines',
                    name="Surgery Start/End",
                    line=dict(color="darkred", dash="dash", width=2),  # Use width > 0 for the legend
                    showlegend=True
                )
            )
    # Layout settings
    legend_opts = dict(
        bgcolor='#f7f7f5',
        bordercolor='gray',
        borderwidth=2,
        font=dict(size=size_legend)
    )
    if xlegend is not None and ylegend is not None:
        legend_opts.update(x=xlegend, y=ylegend)

    fig.update_layout(
        height=height, width=width,
        title_text=f"<b>Vital Signs Time Series for Patient {caseid}</b>",
        title=dict(x=0.5, xanchor='center', yanchor='top'),
        plot_bgcolor='#f7f7f5',
        legend=legend_opts
    )
    fig.update_traces(connectgaps=connectgaps)

    return fig

def plot_multiple_timeseries_mplb(vital_signs_df, caseid, vital_signs, 
                             general_info_df=None, startendline=False,
                             width=12, height=8, xlegend=1.25, ylegend=0.8,
                             size_legend=10, title=True,color_map=None):
    """
        Function to plot multiple time series for a given patient ID using matplotlib. 
        Does the same as plot_multiple_timeseries_plotly but uses matplotlib for plotting.

        Arguments:
        - vital_signs_df: DataFrame containing the vital signs data.
        - caseid: Patient ID to filter the data.
        - vital_signs: List of vital signs to plot.
        - general_info_df: DataFrame containing general information about the patient.
        - startendline: Boolean to indicate if vertical lines for surgery/anesthesia should be plotted.
        - width: Width of the plot.
        - height: Height of the plot.
        - xlegend: X position of the legend.
        - ylegend: Y position of the legend.
        - size_legend: Font size of the legend.
        - title: Boolean to indicate if a title should be added to the plot.
        
        Returns:
        - fig: Matplotlib figure object containing the plot.
    """

    # Filter by patient and convert time to minutes
    df = vital_signs_df[vital_signs_df['caseid'] == caseid].copy()
    df['time'] = df['time'] / 60

    # Remove signals with no useful data
    vital_signs_cleaned = []
    for signal in vital_signs:
        data = df[signal]
        if not (data.isna().all() or data.dropna().eq(0).all()):
            vital_signs_cleaned.append(signal)
        else:
            print(f"No data for {signal}")

    # Define subplot layout
    num_signals = len(vital_signs_cleaned)
    num_cols = int(np.ceil(np.sqrt(num_signals)))
    num_rows = int(np.ceil(num_signals / num_cols))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(width, height))
    axes = np.array(axes).reshape(-1)  # Ensure 1D array for easy indexing

    # Color mapping
    if color_map is None:
        color_map = {
            'sbp': 'teal', 'mbp': 'slateblue', 'dbp': 'lightseagreen',
            'hr': 'navy', 'spo2': 'darkorange', 'bis': 'purple',
            'insp_sevo': 'seagreen', 'exp_sevo': 'yellowgreen'
        }
        

    # Plot blood pressure signals together if present
    bp_signals = {'sbp', 'mbp', 'dbp'}
    bp_included = any(sign in bp_signals for sign in vital_signs_cleaned)

    plot_idx = 0
    for signal in vital_signs_cleaned:
        if signal in bp_signals and bp_included:
            if signal == 'mbp':
                ax = axes[plot_idx]
                for bp in ['sbp', 'mbp', 'dbp']:
                    if bp in df.columns:
                        valid_bp = df[['time', bp]].dropna()
                        if not valid_bp.empty:
                            ax.plot(valid_bp['time'], valid_bp[bp], label=bp,
                                    linestyle='--' if bp == 'dbp' else '-', 
                                    color=color_map[bp])
                ax.set_title("Blood Pressure")
                bp_included = False
                plot_idx += 1
            else:
                continue
        else:
            valid_data = df[['time', signal]].dropna()
            if valid_data.empty:
                print(f"{signal} has only NaNs, skipping.")
                continue
            ax = axes[plot_idx]
            ax.plot(valid_data['time'], valid_data[signal], label=signal, color=color_map[signal])
            ax.set_title(signal)
            plot_idx += 1

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Values")
        ax.grid(axis='y', linestyle='--', color='gray', linewidth=0.5, alpha=0.3)

        # Optional surgery/anesthesia vertical lines
        if startendline and general_info_df is not None:
            gen_info = general_info_df[general_info_df['caseid'] == caseid]
            if not gen_info.empty:
                opstart, opend = gen_info.iloc[0][['opstart', 'opend']] / 60
                anestart, aneend = gen_info.iloc[0][['anestart', 'aneend']] / 60
                ax.axvline(opstart, color='darkred', linestyle='--', 
                           label='Start/End Surgery' if plot_idx == 1 else None)
                ax.axvline(opend, color='darkred', linestyle='--')
                ax.axvline(anestart, color='crimson', linestyle='--',
                            label='Start/End Anesthesia' if plot_idx == 1 else None)
                ax.axvline(aneend, color='crimson', linestyle='--')

    # Remove unused subplots
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    # Add legend and title
    fig.legend(bbox_to_anchor=(xlegend, ylegend), fontsize=size_legend)
    if title:
        fig.suptitle(f"Vital Signs Time Series for Patient {caseid}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    return fig


