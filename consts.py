available_functions= [
    {
        "name": "create_subject",
        "description": "Create a new subject",
        "parameters": {
            "name": "Subject name (required)",
            "category": "Subject category (optional, defaults to 'Uncategorized')"
        }
    },
    {
        "name": "add_component_to_subject",
        "description": "Add a component to an existing subject",
        "parameters": {
            "subject_id": "ID of the subject to add component to (required)",
            "name": "Component name (required)",
            "type": "Component type: int, str, bool, date, Array_type, Array_generic, pair, Array_of_pairs, Array_of_strings, Array_of_booleans, Array_of_dates, Array_of_objects (required)",
            "data": "Component data structure (optional)",
            "is_deletable": "Whether component can be deleted (boolean, optional)"
        }
    },
    {
        "name": "create_connection",
        "description": "Create a connection between two subjects with optional data transfers",
        "parameters": {
            "from_subject_id": "Source subject ID (required)",
            "to_subject_id": "Target subject ID (required)",
            "type": "Connection type (optional, defaults to 'manual')",
            "start_date": "Connection start date (ISO datetime string, optional)",
            "end_date": "Connection end date (ISO datetime string, optional)",
            "done": "Whether connection is completed (boolean, optional)",
            "data_transfers": "Array of data transfer objects (optional). Each object should contain: target_component_id (required), operation (optional, defaults to 'replace'), source_component_id (optional), data_value (optional), details (optional)"
        }
    },
    {
        "name": "create_category",
        "description": "Create a new category",
        "parameters": {
            "name": "Category name (required)",
            "description": "Category description (optional)",
            "color": "Category color (hex code, optional, defaults to '#000000')"
        }
    },
    {
        "name": "create_data_transfer",
        "description": "Create a data transfer operation between components",
        "parameters": {
            "target_component_id": "ID of the target component (required)",
            "operation": "Operation type (replace, add, multiply, toggle, append, etc.)",
            "source_component_id": "ID of the source component (optional)",
            "data_value": "Data value for the operation (optional, used when no source component)",
            "schedule_time": "When to execute the transfer (ISO datetime string, optional)"
        }
    }
]