available_functions=[
            {
                "name": "create_subject",
                "description": "Create a new subject",
                "parameters": {
                    "name": "Subject name (required)",
                    "category": "Subject category (optional, defaults to 'Uncategorized')",
                    "template": "Template name to apply (optional, defaults to None)"
                }
            },
            {
                "name": "add_component_to_subject",
                "description": "Add a component to an existing subject",
                "parameters": {
                    "subject_id": "ID of the subject to add component to (required)",
                    "name": "Component name (required)",
                    "type": "Component type: int, str, bool, date, Array_type, Array_generic, pair, Array_of_pairs, Array_of_strings, Array_of_booleans, Array_of_dates, Array_of_objects (required)",
                    "data": "Component data structure (optional). Data formats by type: int: {'item': 0}, str: {'item': ''}, bool: {'item': false}, date: {'item': 'ISO_date_string'}, pair: {'item': {'key': 'string', 'value': any}, 'type': {'key': 'str', 'value': 'any'}}, Array_type: {'type': 'int'}, Array_generic: {'type': 'any'}, Array_of_pairs: {'type': {'key': 'str', 'value': 'any'}}, Array_of_strings: {'type': 'str'}, Array_of_booleans: {'type': 'bool'}, Array_of_dates: {'type': 'date'}, Array_of_objects: {'type': 'object'}. Array types store items separately in ArrayMetadata.",
                }
            },
            {
                "name": "create_connection",
                "description": "create a connection between two subjects with optional data transfers",
                "parameters": {
                    "from_subject_id": "source subject id (required)",
                    "to_subject_id": "target subject id (required)",
                    "type": "connection type (optional, defaults to 'manual')",
                    "start_date": "connection start date (iso datetime string, optional)",
                    "end_date": "connection end date (iso datetime string, optional)",
                    "done": "whether connection is completed (boolean, optional)",
                    "data_transfers": "array of data transfer objects (optional). each object should contain: target_component_id (required), operation (optional, defaults to 'replace'), source_component_id (optional), data_value (optional), details (optional)"
                }
            },
            {
                "name": "create_category",
                "description": "create a new category",
                "parameters": {
                    "name": "category name (required)",
                    "description": "category description (optional)",
                }
            },
            {
                "name": "create_data_transfer",
                "description": "Create a data transfer operation between components",
                "parameters": {
                    "target_component_id": "ID of the target component (required)",
                    "operation": "Operation type. Valid operations by component type: pair=['update_key', 'update_value'], Array_of_pairs=['append', 'remove_back', 'remove_front', 'delete_at', 'push_at', 'update_pair'], Array types=['append', 'remove_back', 'remove_front', 'delete_at', 'push_at', 'update_at'], scalar types (int/str/bool/date)=['replace', 'add', 'multiply', 'toggle']",
                    "source_component_id": "ID of the source component (optional)",
                    "data_value": "Data value for the operation (optional, used when no source component). Format depends on operation: replace={'item': value}, update_key={'item': {'key': new_key}}, update_value={'item': {'value': new_value}}, append={'item': value_to_append}, update_pair={'item': {'key': key, 'value': value}}, update_at={'item': value, 'index': position}, push_at={'item': value, 'index': position}, delete_at={'index': position}",
                    "schedule_time": "When to execute the transfer (ISO datetime string, optional)"
                }
            },
            {
                "name": "create_widget",
                "description": "create a new widget in a subject",
                "parameters": {
                    "name": "widget name (required)",
                    "type": "widget type: daily_todo, table, note, calendar, text_field, component_reference (required)",
                    "host_subject_id": "ID of the subject to host the widget (required)",
                    "reference_component_id": "ID of component to reference (optional, for component_reference widgets)",
                    "data": "Widget-specific data structure (optional)"
                }
            },
            {
                "name": "create_custom_template",
                "description": "Create a reusable template",
                "parameters": {
                    "name": "Template name (required)",
                    "data": "Template structure with components and widgets (required)",
                    "description": "Template description (optional)",
                    "category": "Template category (optional, defaults to 'Uncategorized')"
                }
            },
            {
                "name": "add_todo_to_widget",
                "description": "Add a todo item to a daily_todo widget",
                "parameters": {
                    "widget_id": "ID of the daily_todo widget (required)",
                    "text": "Todo text (required)",
                    "date": "Todo date in YYYY-MM-DD format (optional, defaults to today)",
                    "completed": "Whether todo is completed (boolean, optional)"
                }
            },
            {
                "name": "update_component_data",
                "description": "Update component data or name",
                "parameters": {
                    "component_id": "ID of the component to update (required)",
                    "data": "New component data (optional). Must match component type structure: scalar types need {'item': value}, pair needs {'item': {'key': string, 'value': any}}, array types need {'type': element_type} and items are managed separately",
                    "name": "New component name (optional)"
                }
            },
            {
                "name": "get_subject_full_data",
                "description": "Get complete data for a subject including all components and widgets",
                "parameters": {
                    "subject_id": "ID of the subject (required)"
                }
            },
            {
                "name": "remove_ai_accessible_subject",
                "description": "Remove a subject from the AI-accessible list",
                "parameters": {
                    "subject_id": "ID of the subject to remove from AI-accessible list (required)"
                }
            },
            {
                "name": "delete_subject",
                "description": "Delete a subject and all its components and widgets",
                "parameters": {
                    "subject_id": "ID of the subject to delete (required)"
                }
            },
            {
                "name": "delete_component",
                "description": "Delete a component and remove it from its host subject",
                "parameters": {
                    "component_id": "ID of the component to delete (required)"
                }
            },
            {
                "name": "delete_widget",
                "description": "Delete a widget and associated data",
                "parameters": {
                    "widget_id": "ID of the widget to delete (required)"
                }
            },
            {
                "name": "delete_connection",
                "description": "Delete a connection between subjects",
                "parameters": {
                    "connection_id": "ID of the connection to delete (required)"
                }
            },
            {
                "name": "delete_category",
                "description": "Delete a category and set associated subjects to 'Uncategorized'",
                "parameters": {
                    "name": "Name of the category to delete (required)"
                }
            },
            {
                "name": "delete_data_transfer",
                "description": "Delete a data transfer operation",
                "parameters": {
                    "transfer_id": "ID of the data transfer to delete (required)"
                }
            },
            {
                "name": "delete_custom_template",
                "description": "Delete a custom template",
                "parameters": {
                    "template_id": "ID of the template to delete (required)"
                }
            },
            {
                "name": "delete_todo",
                "description": "Delete a todo item",
                "parameters": {
                    "todo_id": "ID of the todo to delete (required)",
                    "widget_id": "ID of the widget (optional, for verification)"
                }
            },
            {
                "name": "delete_notification",
                "description": "Delete a notification",
                "parameters": {
                    "notification_id": "ID of the notification to delete (required)"
                }
            },
            {
                "name": "create_habit",
                "description": "Create a new habit subject and optionally add it to the habit tracker",
                "parameters": {
                    "name": "Habit name (required)",
                    "description": "Habit description (optional)",
                    "frequency": "Habit frequency like 'Daily', 'Weekly' (optional, defaults to 'Daily')",
                    "add_to_tracker": "Whether to add to habit tracker (boolean, optional, defaults to true)"
                }
            },
            {
                "name": "add_habit_to_tracker",
                "description": "Add an existing habit subject to the habit tracker",
                "parameters": {
                    "habit_id": "ID of the habit subject to add (required)"
                }
            },
            {
                "name": "remove_habit_from_tracker",
                "description": "Remove a habit from the habit tracker (does not delete the habit subject)",
                "parameters": {
                    "habit_id": "ID of the habit subject to remove (required)"
                }
            },
            {
                "name": "mark_habit_complete",
                "description": "Mark a habit as completed or not completed for today",
                "parameters": {
                    "habit_id": "ID of the habit subject (required)",
                    "completed": "Whether habit is completed (boolean, required)"
                }
            },
            {
                "name": "get_daily_habits_status",
                "description": "Get all habits and their completion status for a specific date",
                "parameters": {
                    "date": "Date in YYYY-MM-DD format (optional, defaults to today)"
                }
            },
            {
                "name": "get_habit_tracker_data",
                "description": "Get complete habit tracker data including all habits and their status",
                "parameters": {}
            },
            {
                "name": "get_habit_detailed_status",
                "description": "Get detailed status information for a specific habit including todos and completion percentages",
                "parameters": {
                    "habit_id": "ID of the habit subject (required)",
                    "date": "Date in YYYY-MM-DD format (optional, defaults to today)"
                }
            },
            {
                "name": "get_habits_count",
                "description": "Get the total count of habits in the tracker efficiently without loading all data",
                "parameters": {}
            }
        ]