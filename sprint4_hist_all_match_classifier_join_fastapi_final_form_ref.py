import os
import pandas as pd
import glob
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from enum import Enum
import numpy as np
import shutil
from langchain_openai import OpenAIEmbeddings
import traceback
from typing import List, Optional
from fastapi.responses import FileResponse
from typing import Union

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Set Globals used throughout functions here
vectorstore = None
selected_field = None
bcis_df = None  # Global to store BCIS dataframe
filtered_df = None # Global to store filtered BCIS dataframe
supplier_df = None  # Global to store supplier_df dataframe
supplier_filename = None
supplier_name = None
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
upload_dir = os.getcwd()
data_dir = os.path.join(upload_dir, "data")
accept_rowcount = 0
reject_rowcount = 0
dup_supplier_rowcount = 0
dedup_supplier_rowcount = 0

# Ensure the /data directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ensure the /data/reject directory exists
reject_dir = os.path.join(data_dir, "reject")
if not os.path.exists(reject_dir):
    os.makedirs(reject_dir)

# Ensure the /data/reference directory exists
reference_dir = os.path.join(data_dir, "reference")
if not os.path.exists(reference_dir):
    os.makedirs(reference_dir)


# Generalized function to save any DataFrame and remove previous CSVs with a specific prefix
def save_dataframe_to_csv(data_dir, dataframe, timestamp, prefix):

    # Remove any previous files with the given prefix in the /data directory
    previous_files = glob.glob(os.path.join(data_dir, f"{prefix}_*.csv"))
    for file in previous_files:
        os.remove(file)

    # Save the DataFrame to a new CSV file in the /data directory
    final_file_name = f"{prefix}_{timestamp}.csv"
    final_file_path = os.path.join(data_dir, final_file_name)
    dataframe.to_csv(final_file_path, index=False)
    
    # Return both the file path and file name
    return final_file_path, final_file_name

# Defining the content of the chunks that'll form the vector DB and create vector_build_df
def get_text_chunks_from_csv(csv_df, selected_field):
    chunks = []
    vector_build_data = []  # To store the rows for vector_build_df

    for _, row in csv_df.iterrows():
        if selected_field == 'concat_description':
            content = row['concat_description']
        elif selected_field == 'bcis_price_description':
            content = row['bcis_price_description']
        elif selected_field == 'supplier_description':
            content = row['supplier_description']
        else:
            content = ""
        
        chunks.append(content)
        # Add the content to the vector_build_data (a list of dictionaries)
        vector_build_data.append({'text_chunk': content})
    
    # Create vector_build_df DataFrame
    vector_build_df = pd.DataFrame(vector_build_data)
    
    return chunks, vector_build_df

# Calling the vector DB, currently set to a local vector DB, created locally on the VM/cluster/function app. 
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Handles input to and output from the vector DB
def handle_userinput(user_question, vectorstore, selected_field):
    retriever = vectorstore.as_retriever()
    results = retriever.vectorstore.similarity_search_with_score(user_question)
    
    # Sets the 'vector distance' as a similarity percentage score
    if results:
        doc, score = results[0]
        normalized_score = np.exp(-score)
        normalized_score = normalized_score / max(normalized_score, 1)

        best_match = doc.page_content

        if selected_field == 'concat_description':
            output_field = 'concat_description'
        elif selected_field == 'bcis_price_description':
            output_field = 'bcis_price_description'
        elif selected_field == 'supplier_description':
            output_field = 'existing_supplier_description'  # Rename only the match result
        else:
            raise ValueError(f"Invalid selected_field: {selected_field}")

        return {
            'supplier_description': user_question,  # Keep this as supplier_description from the input
            output_field: best_match.strip(),  # Rename match as existing_supplier_description
            'similarity': f"{normalized_score:.2f}"
        }
    return None

# Joining supplier results from merged_df and final_supplier_df on supplier_description
def join_and_save_final_merged_df(merged_df, final_supplier_df):


    # Clean 'supplier_description' columns in both DataFrames
    merged_df['supplier_description'] = merged_df['supplier_description'].str.strip().str.lower()
    final_supplier_df['supplier_description'] = final_supplier_df['supplier_description'].str.strip().str.lower()

    # Perform the join on 'supplier_description'
    new_final_merged_df = pd.merge(
        merged_df, 
        final_supplier_df[['code', 'supplier_description', 'supplier_classification', 'unit', 'price']], 
        on='supplier_description', how='left'
    )
    
    # Drop the unwanted columns: 'supplier_classification_x', 'unit_x', 'price_x', 'supplier_code'
    new_final_merged_df.drop(columns=['supplier_classification_x', 'unit_x', 'price_x', 'supplier_code'], inplace=True)

    # Rename columns from the right side of the join: 'supplier_classification_y', 'unit_y', 'price_y', 'code'
    new_final_merged_df.rename(columns={
        'supplier_classification_y': 'supplier_classification',
        'unit_y': 'unit',
        'price_y': 'price',
        'code': 'supplier_code'
    }, inplace=True)

    # Add an empty "accept" column
    new_final_merged_df['accept'] = ""

    return new_final_merged_df
   
# Does the join between the lookup and the results coming out of the RAG
def lookup_and_join_results(output_file_name, output_file_path, supplier_name, final_supplier_df, filtered_df):
    global timestamp, selected_field

    # Load the supplier results CSV & output it to the data directory
    supplier_results_df = pd.read_csv(output_file_path)
    
    # Remove leading/trailing whitespaces from column names
    supplier_results_df.columns = supplier_results_df.columns.str.strip()
    filtered_df.columns = filtered_df.columns.str.strip()

    # Determine the join field based on the columns in supplier_results_df
    if 'bcis_price_description' in supplier_results_df.columns:
        join_field = 'bcis_price_description'
    elif 'concat_description' in supplier_results_df.columns:
        join_field = 'concat_description'
    elif 'existing_supplier_description' in supplier_results_df.columns:
        join_field = 'existing_supplier_description'
    else:
        raise HTTPException(status_code=400, detail="No valid join field found in supplier_results_df.")
    
    # Trim leading/trailing spaces from all string columns in filtered_df & supplier_results_df
    filtered_df = filtered_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    supplier_results_df = supplier_results_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    save_dataframe_to_csv(data_dir, supplier_results_df, timestamp, "supplier_results_df")

    # Ensure that filtered_df has the appropriate join field
    if join_field not in filtered_df.columns and join_field != 'existing_supplier_description':
        raise HTTPException(status_code=400, detail=f"Join field '{join_field}' not found in the filtered file.")

    # Perform the join on the corresponding fields
    try:
        if 'existing_supplier_description' in supplier_results_df.columns:
            # Ensure join happens only on supplier_results_df.existing_supplier_description and filtered_df.supplier_description
            merged_df = pd.merge(supplier_results_df, filtered_df, left_on='existing_supplier_description', right_on='supplier_description', how='left')
        else:
            # Handle other join cases
            merged_df = pd.merge(supplier_results_df, filtered_df, on=join_field, how='left')

        save_dataframe_to_csv(data_dir, merged_df, timestamp, "merged_df_0")

        # Drop unwanted columns
        merged_df = merged_df.drop(columns=['supplier_description_y', 'similarity_y'])

        # Rename the necessary columns
        merged_df = merged_df.rename(columns={'supplier_description_x': 'supplier_description', 'similarity_x': 'similarity'})

        save_dataframe_to_csv(data_dir, merged_df, timestamp, "merged_df_1")
        save_dataframe_to_csv(data_dir, final_supplier_df, timestamp, "final_supplier_df_1")

        # Do the final merge and bring back the price & unit of measure from final_supplier_df
        final_merged_df = join_and_save_final_merged_df(merged_df, final_supplier_df)

        save_dataframe_to_csv(data_dir, final_merged_df, timestamp, "final_merged_df_1")

        # Populate the 'supplier' field with the supplier_name
        final_merged_df['supplier'] = supplier_name

        # Add 'similarity' column from supplier_results_df
        final_merged_df['similarity'] = supplier_results_df['similarity']

        # Add an empty "accept" column
        final_merged_df['accept'] = ""

        # Specify the output columns
        output_columns = ['bcis_price_guid', 'concat_description', 'bcis_price_description', 'bcis_class_description',
                          'supplier_description', 'supplier_code', 'supplier_classification', 'unit', 'price', 
                          'similarity', 'bcis_class_string', 'hier1', 'hier2', 'hier3', 'hier4', 'supplier','accept']

        final_merged_df = final_merged_df[output_columns]

        # Create new output file name
        join_output_file_name = f"final_join_bcis_supplier_{timestamp}.csv"
        join_output_file_path = os.path.join(os.getcwd(), join_output_file_name)


        # Save the final merged DataFrame to CSV
        final_merged_df.to_csv(join_output_file_path, index=False)

        # Save the processed DataFrame
        save_dataframe_to_csv(data_dir, final_merged_df, timestamp, "final_merged_df")

        return join_output_file_name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during file lookup and join: {str(e)}")

# Updated process_supplier_file_deduplication function
def process_supplier_file_deduplication(supplier_df, supplier_name_param):
    global timestamp, dup_supplier_rowcount, dedup_supplier_rowcount  # remove supplier_name from globals
    
    # Initialize dup_supplier_df and dedup_supplier_df to handle scenarios with no duplicates
    dup_supplier_df = pd.DataFrame()  # Empty DataFrame for duplicate records
    dedup_supplier_df = supplier_df.copy()  # Assume all records are unique if no duplicates are found
    
    # Default values for filenames when there are no duplicates
    dedup_supplier_df_file_name = None
    dup_supplier_df_file_name = None
    dedup_comment = "The supplier file has no duplicate values of 'supplier_description'."
    
    # Check for duplicates
    if supplier_df['supplier_description'].duplicated().any():
        dedup_comment = "The supplier file has duplicate values of 'supplier_description'."

        # Separate duplicates and unique rows
        dup_supplier_df = supplier_df[supplier_df.duplicated(subset='supplier_description', keep=False)]
        dedup_supplier_df = supplier_df.drop_duplicates(subset='supplier_description', keep=False)

        # Save the processed DataFrame for duplicates
        dup_supplier_df_file_path, dup_supplier_df_file_name = save_dataframe_to_csv(data_dir, dup_supplier_df, timestamp, "dup_supplier_df")
        dedup_supplier_df_file_path, dedup_supplier_df_file_name = save_dataframe_to_csv(data_dir, dedup_supplier_df, timestamp, "dedup_supplier_df")

        # Deduplicate the duplicate rows by selecting the row with the lowest price
        if 'price' in dup_supplier_df.columns:
            deduped_duplicates = dup_supplier_df.sort_values('price').drop_duplicates(subset='supplier_description', keep='first')
        else:
            deduped_duplicates = dup_supplier_df.drop_duplicates(subset='supplier_description', keep='first')

        dedup_supplier_df = pd.concat([dedup_supplier_df, deduped_duplicates])

        # Save the duplicate rows to a CSV file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        dup_output_file_name = f"{supplier_name_param}_duplicates_{timestamp}.csv"  # updated supplier name reference
        dup_output_file_path = os.path.join(os.getcwd(), dup_output_file_name)
        dup_supplier_df.to_csv(dup_output_file_path, index=False)

        dedup_comment += f" Duplicates saved in: {dup_output_file_name}."
    
    else:
        # No duplicates, return the original supplier_df as deduplicated data
        dedup_supplier_df = supplier_df

    # Update row counts
    dup_supplier_rowcount = len(dup_supplier_df)
    dedup_supplier_rowcount = len(dedup_supplier_df)
    
    return dedup_supplier_df, dedup_supplier_df_file_name, dup_supplier_df_file_name, dedup_comment


# Enum for field selection
class FieldOption(str, Enum):
    concat_description = 'concat_description'
    bcis_price_description = 'bcis_price_description'

# Enum for supplier selection
class SupplierName(str, Enum):
    bobs_brick_bodega = 'bobs_brick_bodega'
    travis_perkins = 'travis_perkins'
    jewsons = 'jewsons'
    wickes = 'wickes'
    empty = 'empty'

# Enum for hier2 filter
class HierFilter(str, Enum):
    chimneys_and_fireplaces = "|Chimneys and fireplaces|"
    doors_windows_and_rooflights = "|Doors, windows and rooflights|"
    landscaping_and_fencing_equipment = "|Landscaping and fencing equipment|"
    proofing_and_jointing = "|Proofing and jointing|"
    site_and_external_works = "|Site and external works|"
    disposal_systems = "|Disposal systems|"
    underground_drainage_and_services = "|Underground drainage and services|"
    cement_based_materials = "|Cement based materials|"
    vermin_control = "|Vermin control|"
    building_chemicals = "|Building chemicals|"
    epoxy_resin_mortars_and_grout = "|Epoxy resin mortars and grout|"
    architectural_metalwork = "|Architectural metalwork|"
    building_services = "|Building services|"
    fixings_and_adhesives = "|Fixings & adhesives|"
    finishes_and_painting = "|Finishes and painting|"
    stair_components = "|Stair components|"
    access_equipment = "|Access equipment|"
    structural_metalwork = "|Structural metalwork|"
    plaster_and_render = "|Plaster and render|"
    insulation = "|Insulation|"
    structural_precast_concrete = "|Structural precast concrete|"
    builders_metalwork = "|Builders metalwork|"
    lifts_escalators_conveyors_and_hoists = "|Lifts, escalators, conveyors and hoists|"
    sanitary_fittings = "|Sanitary fittings|"
    panel_linings_and_partitions = "|Panel linings and partitions|"
    prefabricated_structures = "|Prefabricated structures|"
    ironmongery = "|Ironmongery|"
    timber_and_building_boards = "|Timber and building boards|"
    concrete_accessories = "|Concrete accessories|"
    roofing_and_cladding = "|Roofing and cladding|"
    concrete_placing_and_finishing_equipment = "|Concrete placing and finishing equipment|"
    masonry = "|Masonry|"
    glass_and_glazing = "|Glass and glazing|"
    aggregates = "|Aggregates|"
    stairs_and_balustrades = "|Stairs, and balustrades|"
    furniture_fixtures_and_fittings = "|Furniture fixtures and fittings|"
    mouldings_edgings_and_trims = "|Mouldings, edgings and trims|"

# Helper function to parse strings to HierFilter Enums
def parse_hier_filter_values(hier_filter_values: Optional[List[str]]):
    
    #Converts a list of strings to HierFilter Enums, adding `|` characters as needed.
    #Skips any empty or placeholder values.
    
    cleaned_values = []
    for value in hier_filter_values or []:
                # Skip empty strings
        if not value.strip():
            continue
        # Add `|` characters and strip whitespace
        stripped_value = f"|{value.strip()}|"
        # Match the cleaned value with an Enum if it exists
        try:
            cleaned_values.append(HierFilter(stripped_value))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid hier_filter_values input: {value}")
    return cleaned_values

# Function to filter the DataFrame based on hier2 values
def filter_bcis_df_by_hier(bcis_df, hier_filter_values):
    if not hier_filter_values:
        # Explicitly assign the full dataframe
        return bcis_df.copy()  # Use `copy()` to avoid any reference issues
    # Apply filters if hier_filter_values exist
    selected_hier_values = [hier.value for hier in hier_filter_values]
    bcis_hier_df = bcis_df[bcis_df['hier2'].isin(selected_hier_values)]
    return bcis_hier_df

# New endpoint to get historical supplier material loads
@app.get("/get-historical-supplier-loads/")
async def get_historical_supplier_loads():
    file_name = "historical_supplier_material_loads.csv"
    file_path = os.path.join(os.getcwd(), file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Historical supplier loads file does not exist.")
    
    try:
        historical_data_df = pd.read_csv(file_path)
        return JSONResponse(content=historical_data_df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading historical supplier loads: {str(e)}")

# Define an Enum for match type
class MatchType(str, Enum):
    full_match = "Full match"
    cross_supplier = "Cross supplier"
    supplier_specific = "Supplier specific"

# Define required columns in the accepted mapping file
ACCEPTED_REQUIRED_COLUMNS = [
    'bcis_price_guid', 'concat_description', 'bcis_price_description', 'bcis_class_description', 
    'supplier_description', 'supplier_code', 'supplier_classification', 'unit', 'price', 
    'similarity', 'bcis_class_string', 'hier1', 'hier2', 'hier3', 'hier4', 'supplier', 'accept'
]

# The main function to handle file upload, filtering (including by hierarchy) and vector DB creation
# Have updated this FastAPI endpoint to accept strings and parse enums using the helper function so Hier filtering works
from typing import Union

@app.post("/upload-bcis-and-select-field/")
async def upload_bcis_and_select_field(
    file: Union[UploadFile, str, None] = None,  # Union type to accommodate empty string input
    match_type: MatchType = Form(...),
    field_choice: FieldOption = Form(None),
    specific_supplier: SupplierName = Form(SupplierName.empty),
    hier_filter_values: Optional[List[str]] = Form(None)
):
    global vectorstore, selected_field, bcis_df, filtered_df, timestamp
    selected_field = field_choice.value if field_choice else None

    # Parse and convert hier_filter_values strings to HierFilter Enums, skip if empty
    if hier_filter_values and any(val.strip() for val in hier_filter_values):
        hier_filter_values = parse_hier_filter_values(hier_filter_values)
    else:
        hier_filter_values = None  # Set as None if no valid values to avoid further processing

    # Treat empty string as None for file
    if file == "" or file is None:
        file = None
    elif isinstance(file, str):  # This handles cases where file might still be passed as a string
        file = None

    # Use the uploaded file if provided; otherwise, use the default file
    if file:
        if isinstance(file, UploadFile) and file.content_type != "text/csv":
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
        
        file_path = os.path.join(upload_dir, file.filename)
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}")
    else:
        # Use the reference file if no file uploaded
        file_path = os.path.join(reference_dir, "bcis_price_class_hier.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="No file provided and default BCIS file does not exist.")

    # Define the required columns
    required_columns = [
        'bcis_price_guid', 'concat_description', 'bcis_price_description',
        'bcis_class_description', 'supplier_description', 'supplier_code', 
        'supplier_classification', 'unit', 'price', 'similarity', 
        'bcis_class_string', 'hier1', 'hier2', 'hier3', 'hier4', 'supplier', 'effective_from', 'effective_to' 
    ]

    # Read and process the file
    try:
        bcis_df = pd.read_csv(file_path)
        save_dataframe_to_csv(data_dir, bcis_df, timestamp, "bcis_df_0")

        # Convert all columns to strings
        bcis_df = bcis_df.astype(str)

        # 1. Check if required columns exist
        if not all(col in bcis_df.columns for col in required_columns):
            missing_cols = set(required_columns) - set(bcis_df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns in the BCIS file: {', '.join(missing_cols)}")
        
        # 2. Filter the DataFrame based on hier_filter_values
        bcis_hier_df = filter_bcis_df_by_hier(bcis_df, hier_filter_values)
        save_dataframe_to_csv(data_dir, bcis_hier_df, timestamp, "bcis_hier_df_0")

        # Ensure all columns are strings for consistency
        bcis_hier_df['similarity'] = bcis_hier_df['similarity'].astype(str)
        bcis_hier_df['supplier'] = bcis_hier_df['supplier'].astype(str)

        # 3. Handle match type selection
        if match_type == MatchType.full_match:
            save_dataframe_to_csv(data_dir, bcis_hier_df, timestamp, "pre_full_match_bcis_df")
            filtered_df = bcis_hier_df[(bcis_hier_df['supplier'] == 'bcis')]
            filtered_df = filtered_df.drop(columns=['effective_from', 'effective_to'], errors='ignore')
            save_dataframe_to_csv(data_dir, filtered_df, timestamp, "post_full_match_filtered_df")
            if len(filtered_df) < 9208:
                raise HTTPException(status_code=400, detail=f"Row count of filtered data is less than 9208. Found: {len(filtered_df)}")
            
            # Let user choose between concat_description and bcis_price_description
            if not selected_field:
                raise HTTPException(status_code=400, detail="Please choose a field for the vector database.")

        elif match_type == MatchType.cross_supplier:
            save_dataframe_to_csv(data_dir, bcis_hier_df, timestamp, "pre_cross_supplier_bcis_df")
            # Filter for supplier <> 'bcis' and similarity='1'
            filtered_df = bcis_hier_df[(bcis_hier_df['supplier'] != 'bcis')]
            filtered_df = filtered_df.drop(columns=['effective_from', 'effective_to'], errors='ignore')
            selected_field = 'supplier_description'  # Use supplier_description for cross-supplier
            save_dataframe_to_csv(data_dir, filtered_df, timestamp, "post_cross_supplier_filtered_df")

            # Handle empty filtered results
            if filtered_df.empty:
                raise HTTPException(status_code=400, detail="No data found for cross-supplier matching.")

        elif match_type == MatchType.supplier_specific:
            save_dataframe_to_csv(data_dir, bcis_hier_df, timestamp, "pre_supplier_specific_bcis_df")
            
            # Filter for specific supplier with similarity='1'
            filtered_df = bcis_hier_df[(bcis_hier_df['supplier'] == specific_supplier)]
            filtered_df = filtered_df.drop(columns=['effective_from', 'effective_to'], errors='ignore')
            selected_field = 'supplier_description'  # Use supplier_description for specific supplier
            save_dataframe_to_csv(data_dir, filtered_df, timestamp, "post_supplier_specific_filtered_df")

            # Handle empty filtered results
            if filtered_df.empty:
                raise HTTPException(status_code=400, detail=f"No data found for supplier: {specific_supplier}")

        else:
            raise HTTPException(status_code=400, detail="Invalid match type selected.")

        # Check if the selected field exists in the uploaded file
        if selected_field not in bcis_df.columns:
            raise HTTPException(status_code=400, detail=f"Selected field '{selected_field}' not found in the CSV file.")

        # 4. Build the vector database and populate vector_build_df
        text_chunks, vector_build_df = get_text_chunks_from_csv(filtered_df, selected_field)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No valid data found in the selected field.")

        vectorstore = get_vectorstore(text_chunks)

        # Save the filtered_df and vector_build_df to CSVs
        bcis_df_file_path, bcis_df_file_name = save_dataframe_to_csv(data_dir, bcis_df, timestamp, "bcis_df")
        filtered_df_file_path, filtered_df_file_name = save_dataframe_to_csv(data_dir, filtered_df, timestamp, "filtered_df")
        vector_build_df_file_path, vector_build_df_file_name = save_dataframe_to_csv(data_dir, vector_build_df, timestamp, "vector_build_df")

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty or invalid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the CSV file: {str(e)}")

    return {
        "message": "BCIS file uploaded, filtered, and vector database created",
        "uploaded_file_path": bcis_df_file_path,
        "uploaded_file_name": bcis_df_file_name,
        "filtered_file_path": filtered_df_file_path,
        "filtered_file_name": filtered_df_file_name,
        "vector_build_file_path": vector_build_df_file_path,
        "vector_build_file_name": vector_build_df_file_name,
        "selected_field": selected_field
    }


# Appends processing meta data to the historical_supplier_material_loads.csv
# Modified append_to_historical_loads function to include new parameters
# Updated append_to_historical_loads function
def append_to_historical_loads(file_name, supplier_name_rej, supplier_row_count, dup_supplier_row_count, unique_supplier_row_count, accept_rowcount, reject_rowcount):
    historical_file_path = os.path.join(os.getcwd(), 'historical_supplier_material_loads.csv')

    # New data structure with the additional fields
    new_row = pd.DataFrame({
        'supplier file name': [file_name],
        'Supplier name': [supplier_name_rej],  # updated to supplier_name_rej
        'supplier file size': [supplier_row_count],
        'supplier duplicates': [dup_supplier_row_count],
        'supplier unique records': [unique_supplier_row_count],
        'supplier classifications accepted': [accept_rowcount],
        'supplier classifications rejected': [reject_rowcount],
        'file processed timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'username': ['A Churley'],
        'status': ['Processing'],
        'actions': ['Review']
    })

    try:
        if os.path.exists(historical_file_path):
            historical_df = pd.read_csv(historical_file_path)
            historical_df.columns = historical_df.columns.str.strip()
            historical_df = historical_df.loc[:, ~historical_df.columns.duplicated()]
            updated_df = pd.concat([historical_df, new_row], ignore_index=True)
        else:
            updated_df = new_row

        updated_df.to_csv(historical_file_path, index=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating historical supplier loads: {str(e)}")


# Upload supplier materials file
@app.post("/upload-supplier/")
async def upload_supplier_file(file: UploadFile):
    global vectorstore, selected_field, timestamp, supplier_filename, supplier_df 

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")

    supplier_filename = file.filename
    supplier_name = supplier_filename.rsplit('_', 1)[0]

    valid_suppliers = {'bobs_brick_bodega', 'travis_perkins', 'jewsons', 'wickes'}
    if supplier_name not in valid_suppliers:
        raise HTTPException(status_code=400, detail=f"Invalid supplier name. Valid options are {', '.join(valid_suppliers)}.")

    # Define the upload directory
    upload_dir = os.getcwd()

    # Perform cleanup: Remove any previous files with the <supplier_name>_*.csv pattern
    previous_files = glob.glob(os.path.join(upload_dir, f"{supplier_name}_*.csv"))
    for old_file in previous_files:
        os.remove(old_file)

    # Proceed with new file <supplier_name>_YYYYMMDDhhmmss.csv creation
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_name = f"{supplier_name}_{timestamp}.csv"
    file_path = os.path.join(upload_dir, output_file_name)

    # Correct file handling here using the file.file attribute
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # prompt file removal
    prompt_files = glob.glob(os.path.join(upload_dir, "prompt_bcis_*.csv"))
    for prompt_file in prompt_files:
        os.remove(prompt_file)

     # join file removal
    join_files = glob.glob(os.path.join(upload_dir, "final_join_bcis*.csv"))
    for join_file in join_files:
        os.remove(join_file)

    if vectorstore is None or selected_field is None:
        raise HTTPException(status_code=400, detail="Vector database not created. Please upload BCIS file and select a field first.")

    try:
        supplier_df = pd.read_csv(file_path)
        save_dataframe_to_csv(data_dir, supplier_df, timestamp, "supplier_df")
        if 'supplier_description' not in supplier_df.columns:
            raise HTTPException(status_code=400, detail="Supplier file CSV does not contain 'supplier_description' column.")

              
        # Call the deduplication function here
        dedup_supplier_df, dedup_supplier_df_file_name, dup_supplier_df_file_name, suppliers_file_deduplication_comment = process_supplier_file_deduplication(supplier_df, supplier_name)


        # Check if dedup_supplier_df exists and is not empty, set final_supplier_df to dedup_supplier_df, else set final_supplier_df to supplier_df
        if dedup_supplier_df is not None and not dedup_supplier_df.empty:
            final_supplier_df = dedup_supplier_df
        else:
            final_supplier_df = supplier_df     

        save_dataframe_to_csv(data_dir, final_supplier_df, timestamp, "final_supplier_df")

        output_file_name = f"prompt_bcis_supplier_{timestamp}.csv"
        output_file_path = os.path.join(os.getcwd(), output_file_name)

        result_list = []
        for _, row in final_supplier_df.iterrows():
            user_question = row['supplier_description']
            result = handle_userinput(user_question, vectorstore, selected_field)
            if result:
                result_list.append(result)
        
        df = pd.DataFrame(result_list)
        df.to_csv(output_file_path, mode='w', header=True, index=False)

        # Call lookup_and_join_results with final_supplier_df and filtered_df
        join_output_file_name = lookup_and_join_results(output_file_name, output_file_path, supplier_name, final_supplier_df, filtered_df)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during classification: {str(e)}")
    
    return {
        "message": "CSV processing completed",
        "prompt_output_file": output_file_name,
        "join_output_file": join_output_file_name,
        "suppliers_file_deduplication_comment": suppliers_file_deduplication_comment,
        "dup_supplier_df_file_name": dup_supplier_df_file_name,
        "dedup_supplier_df_file_name": dedup_supplier_df_file_name
         }
         
@app.get("/get-file-results/")
async def get_file_results(file: str):
    """
    Retrieve and display the file content as JSON.
    """
    if not os.path.exists(file):
        raise HTTPException(status_code=400, detail="File does not exist.")
    
    try:
        # Load the CSV and convert to JSON records
        result_df = pd.read_csv(file)
        records = []
        for record in result_df.to_dict(orient="records"):
            try:
                # Handle NaNs and other special values by converting them to strings
                record = {k: (v if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v) else str(v)) for k, v in record.items()}
                records.append(record)
            except Exception as inner_e:
                raise HTTPException(status_code=500, detail=f"Error processing record: {str(inner_e)}")
        
        return JSONResponse(content=records)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")

@app.get("/download-file/")
async def download_file(file: str):
    """
    Download the file as a CSV.
    """
    if not os.path.exists(file):
        raise HTTPException(status_code=400, detail="File does not exist.")
    
    return FileResponse(path=file, media_type="text/csv", filename=os.path.basename(file))


#after the user has accepted or rejected the output from semantic matching the file final_merged_df_{timestamp}_accept.csv is uploaded 
#for the accepted mappings final_merged_accept_df_YYYYMMDDhhmmss.csv is generated 
#for the rejected mappings {supplier_name}_YYYYMMDDhhmmss.csv is generated and the rejected mapping can be subjected to other matching strategies
# Modified process_final_merged_accept function to include the new fields and call append_to_historical_loads
@app.post("/process-final-merged-accept/")
async def process_final_merged_accept(file: UploadFile):
    global accept_rowcount, reject_rowcount, dup_supplier_rowcount, dedup_supplier_rowcount

    try:
        if file.content_type != 'text/csv':
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
        file_path = os.path.join(data_dir, file.filename)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        final_merged_accept_reject_df = pd.read_csv(file_path)

        # Required columns check
        required_columns = [
            'bcis_price_guid', 'concat_description', 'bcis_price_description', 'bcis_class_description',
            'supplier_description', 'supplier_code', 'supplier_classification', 'unit', 'price',
            'similarity', 'bcis_class_string', 'hier1', 'hier2', 'hier3', 'hier4', 'supplier', 'accept'
        ]
        missing_columns = [col for col in required_columns if col not in final_merged_accept_reject_df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns in the file: {', '.join(missing_columns)}")

        final_merged_accept_reject_df['accept'] = final_merged_accept_reject_df['accept'].astype(str)
        final_merged_accept_reject_df.loc[final_merged_accept_reject_df['accept'] == "1", 'similarity'] = 1

        # Separate accepted and rejected DataFrames & remove the 'accept' column from accepted_df
        accepted_df = final_merged_accept_reject_df[final_merged_accept_reject_df['accept'] == "1"].copy()
        accepted_df.drop(columns=['accept'], inplace=True)
        rejected_df = final_merged_accept_reject_df[final_merged_accept_reject_df['accept'] == "0"][
            ['supplier_code', 'supplier_description', 'supplier_classification', 'unit', 'price']
        ].rename(columns={'supplier_code': 'code'}).copy()

        # Assign row counts
        accept_rowcount = len(accepted_df)
        reject_rowcount = len(rejected_df)
        supplier_name_rej = supplier_filename.rsplit('_', 1)[0]

        # Save accepted records
        if not accepted_df.empty:
            today_str = datetime.now().strftime("%Y%m%d")
            accepted_df['effective_from'] = today_str
            accepted_df['effective_to'] = '20991231'
            accepted_file_path, _ = save_dataframe_to_csv(data_dir, accepted_df, datetime.now().strftime("%Y%m%d%H%M%S"), "final_merged_accept_df")
        else:
            accepted_file_path = "No accepted records to save."

        # Save rejected records in the reject directory without the 'supplier' column
        rejected_files = []
        if not rejected_df.empty:
            # Write each rejected file per supplier group in reject directory
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            reject_file_path, _ = save_dataframe_to_csv(reject_dir, rejected_df, timestamp, supplier_name_rej)
            rejected_files.append(reject_file_path)
        else:
            rejected_files.append("No rejected records to save.")

        # Append accepted_df to bcis_df and save as bcis_price_class_hier.csv
        bcis_file_path = os.path.join(reference_dir, "bcis_price_class_hier.csv")

        # Archive existing bcis_price_class_hier.csv if it exists
        if os.path.exists(bcis_file_path):
            archive_file_name = f"bcis_price_class_hier_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            archive_file_path = os.path.join(reference_dir, archive_file_name)
            os.rename(bcis_file_path, archive_file_path)

        # Append accepted_df to bcis_df and save new file
        combined_bcis_df = pd.concat([bcis_df, accepted_df], ignore_index=True)
        combined_bcis_df.to_csv(bcis_file_path, index=False)

        # Format response paths
        accepted_file_response = os.path.basename(accepted_file_path)
        rejected_files_response = (
            os.path.basename(rejected_files[0]) if len(rejected_files) == 1 else "No rejected records to save"
            if not rejected_files else ", ".join([os.path.basename(f) for f in rejected_files])
        )

        # Call append_to_historical_loads with all updated values
        append_to_historical_loads(
            file_name=supplier_filename,
            supplier_name_rej=supplier_name_rej,
            supplier_row_count=len(supplier_df),
            dup_supplier_row_count=dup_supplier_rowcount,
            unique_supplier_row_count=dedup_supplier_rowcount,
            accept_rowcount=accept_rowcount,
            reject_rowcount=reject_rowcount
        )

        return {
            "message": "File processed successfully",
            "accepted_file": accepted_file_response,
            "accepted_rowcount": accept_rowcount,
            "rejected_file": rejected_files_response,
            "rejected_rowcount": reject_rowcount,
            "combined_bcis_file": os.path.basename(bcis_file_path)  # Return the final combined BCIS file name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")