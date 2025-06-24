from models import User, DataTransfer_db, DataTransfer, Subject_db, Subject, Component_db,  Connection_db, Connection, Widget, Widget_db, Todo_db, Todo, Category_db,  Arrays, CustomTemplate_db , Notification_db , NotificationCount , TEMPLATES
from datetime import datetime, timedelta, timezone
from utils.habit_tracker import HabitTrackerManager
from models.subject import Subject
from models.arrayItem import Arrays

async def execute_function_call(function_name: str, arguments: dict, user: User):
    """
    Execute function calls requested by the AI
    """
    try:
        if function_name == "create_subject":
            subject = Subject(
                name=arguments.get("name", ""),
                owner=user.id,
                template=arguments.get("template", ""),
                category=arguments.get("category", "Uncategorized")
            )
            subject.save_to_db()
            return {"success": True, "subject_id": str(subject.id), "message": f"Created subject: {subject.name}"}
        
        elif function_name == "add_component_to_subject":
            subject_id = arguments.get("subject_id")
            
            # Verify subject exists and belongs to user
            subject_db = Subject_db.objects(id=subject_id, owner=user.id).first()
            if not subject_db:
                return {"success": False, "message": "Subject not found or access denied"}
            
            # Load the subject using helper class
            subject = Subject.from_db(subject_db)
            if not subject:
                return {"success": False, "message": "Error loading subject"}
            
            # Add component using the helper method
            component = await subject.add_component(
                component_name=arguments.get("name", ""),
                component_type=arguments.get("type", "str"),
                data=arguments.get("data", None),
            )
            
            if component:
                return {"success": True, "component_id": str(component.id), "message": f"Added component to subject: {component.name}"}
            else:
                return {"success": False, "message": "Failed to create component"}
        
        elif function_name == "create_connection":
            # Create a new connection - fix field names
            from_subject_id = arguments.get("from_subject_id")
            to_subject_id = arguments.get("to_subject_id")
            
            # Verify both subjects exist and belong to user
            from_subject = Subject_db.objects(id=from_subject_id, owner=user.id).first()
            to_subject = Subject_db.objects(id=to_subject_id, owner=user.id).first()
            
            if not from_subject or not to_subject:
                return {"success": False, "message": "One or both subjects not found or access denied"}
            
            # Create connection using helper class with correct parameters
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")

            if not start_date:
                start_date_dt = datetime.utcnow()
            else:
                start_date_dt = datetime.fromisoformat(start_date)
            if not end_date:
                end_date_dt = start_date_dt + timedelta(hours=1)
            else:
                end_date_dt = datetime.fromisoformat(end_date)

            connection = Connection(
                source_subject=from_subject,
                target_subject=to_subject,
                con_type=arguments.get("type", "manual"),
                owner=user.id,
                start_date=start_date_dt.isoformat(),
                end_date=end_date_dt.isoformat(),
                done=arguments.get("done", False)
            )
            
            # Add data transfers if provided
            data_transfers = arguments.get("data_transfers", [])
            if data_transfers:
                for transfer in data_transfers:
                    source_component_id = transfer.get("source_component_id")
                    target_component_id = transfer.get("target_component_id")
                    
                    if not target_component_id:
                        return {"success": False, "message": "Target component is required for data transfer"}
                    
                    # Verify target component exists and belongs to user
                    target_component = Component_db.objects(id=target_component_id, owner=user.id).first()
                    if not target_component:
                        return {"success": False, "message": f"Target component {target_component_id} not found or access denied"}
                    
                    # Verify source component if provided
                    source_component = None
                    if source_component_id:
                        source_component = Component_db.objects(id=source_component_id, owner=user.id).first()
                        if not source_component:
                            return {"success": False, "message": f"Source component {source_component_id} not found or access denied"}
                    
                    # Add data transfer to connection using the helper method
                    await connection.add_data_transfer(
                        source_component=source_component,
                        target_component=target_component,
                        data_value=transfer.get("data_value"),
                        operation=transfer.get("operation", "replace"),
                        details=transfer.get("details")
                    )
            
            # Save connection to database
            connection.save_to_db()
            return {
                "success": True, 
                "connection_id": str(connection.id), 
                "message": f"Created connection between subjects with {len(data_transfers)} data transfers"
            }
        
        elif function_name == "create_category":
            # Create a new category - correct field names
            category_data = {
                "name": arguments.get("name", ""),
                "description": arguments.get("description", ""),
                "owner": user.id
            }
            category_db = Category_db(**category_data)
            category_db.save()
            return {"success": True, "category_id": str(category_db.id), "message": f"Created category: {category_data['name']}"}
        
        elif function_name == "create_data_transfer":
            # Create a new data transfer
            source_component_id = arguments.get("source_component_id")
            target_component_id = arguments.get("target_component_id")
            operation = arguments.get("operation", "replace")
            data_value = arguments.get("data_value")
            schedule_time = arguments.get("schedule_time")
            
            # Verify target component exists and belongs to user
            target_component = Component_db.objects(id=target_component_id, owner=user.id).first()
            if not target_component:
                return {"success": False, "message": "Target component not found or access denied"}
            
            # Verify source component if provided
            if source_component_id:
                source_component = Component_db.objects(id=source_component_id, owner=user.id).first()
                if not source_component:
                    return {"success": False, "message": "Source component not found or access denied"}
            
            # Create data transfer
            data_transfer = DataTransfer(
                source_component=source_component_id,
                target_component=target_component_id,
                data_value=data_value,
                operation=operation,
                owner=user.id,
            )
            
            # Execute immediately if no schedule_time provided or if schedule_time is in the past/now
            if not schedule_time or (data_transfer.schedule_time and datetime.now(timezone.utc) >= data_transfer.schedule_time):
                if data_transfer.execute():
                    return {
                        "success": True, 
                        "data_transfer_id": data_transfer.id, 
                        "message": f"Data transfer executed immediately: {operation} operation on {target_component.name}"
                    }
                else:
                    return {"success": False, "message": "Failed to execute data transfer"}
            else:
                # Save for future execution
                data_transfer.save_to_db()
                return {
                    "success": True, 
                    "data_transfer_id": data_transfer.id, 
                    "message": f"Data transfer scheduled for {data_transfer.schedule_time.isoformat()}: {operation} operation on {target_component.name}"
                }
        
        elif function_name == "create_widget":
            # Create a new widget
            widget_name = arguments.get("name", "")
            widget_type = arguments.get("type", "")
            host_subject_id = arguments.get("host_subject_id", "")
            reference_component_id = arguments.get("reference_component_id")
            widget_data = arguments.get("data", {})
            
            if not widget_name or not widget_type or not host_subject_id:
                return {"success": False, "message": "Name, type, and host_subject_id are required"}
            
            # Verify host subject exists and belongs to user
            host_subject = Subject_db.objects(id=host_subject_id, owner=user.id).first()
            if not host_subject:
                return {"success": False, "message": "Host subject not found or access denied"}
            
            # Verify reference component if provided
            reference_component = None
            if reference_component_id:
                reference_component = Component_db.objects(id=reference_component_id, owner=user.id).first()
                if not reference_component:
                    return {"success": False, "message": "Reference component not found or access denied"}
            
            # Create widget
            widget = Widget(
                name=widget_name,
                widget_type=widget_type,
                host_subject=host_subject_id,
                reference_component=reference_component_id,
                data=widget_data,
                owner=user.id
            )
            widget.save_to_db()
            return {"success": True, "widget_id": str(widget.id), "message": f"Created widget: {widget_name}"}
        
        elif function_name == "create_custom_template":
            # Create a custom template
            template_name = arguments.get("name", "")
            template_data = arguments.get("data", {})
            description = arguments.get("description", "")
            category = arguments.get("category", "Uncategorized")
            
            if not template_name or not template_data:
                return {"success": False, "message": "Name and data are required"}
            
            template = CustomTemplate_db(
                owner=user.id,
                name=template_name,
                description=description,
                data=template_data,
                category=category
            )
            template.save()
            return {"success": True, "template_id": str(template.id), "message": f"Created template: {template_name}"}
        
        elif function_name == "add_todo_to_widget":
            # Add a todo item to a daily_todo widget
            widget_id = arguments.get("widget_id", "")
            todo_text = arguments.get("text", "")
            todo_date = arguments.get("date")
            completed = arguments.get("completed", False)
            
            if not widget_id or not todo_text:
                return {"success": False, "message": "Widget ID and todo text are required"}
            
            # Verify widget exists and belongs to user
            widget = Widget_db.objects(id=widget_id, owner=user.id).first()
            if not widget:
                return {"success": False, "message": "Widget not found or access denied"}
            
            if widget.widget_type != "daily_todo":
                return {"success": False, "message": "This function only works with daily_todo widgets"}
            
            # Use current date if not provided
            if not todo_date:
                todo_date = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Create todo
            todo = Todo(
                text=todo_text,
                completed=completed,
                date=datetime.strptime(todo_date, "%Y-%m-%d"),
                widget_id=widget_id,
                owner=user.id
            )
            todo.save_to_db()
            return {"success": True, "todo_id": str(todo.id), "message": f"Added todo: {todo_text}"}
        
        elif function_name == "update_component_data":
            # Update component data
            component_id = arguments.get("component_id", "")
            new_data = arguments.get("data")
            new_name = arguments.get("name")
            
            if not component_id:
                return {"success": False, "message": "Component ID is required"}
            
            # Verify component exists and belongs to user
            component = Component_db.objects(id=component_id, owner=user.id).first()
            if not component:
                return {"success": False, "message": "Component not found or access denied"}
            
            # Update fields
            updated = False
            if new_data is not None:
                component.data = new_data
                updated = True
            if new_name:
                component.name = new_name
                updated = True
            
            if not updated:
                return {"success": False, "message": "No data provided to update"}
            
            component.save()
            return {"success": True, "component_id": str(component.id), "message": f"Updated component: {component.name}"}
        
        elif function_name == "get_subject_full_data":
            # Get full data for a subject (components and widgets)
            subject_id = arguments.get("subject_id", "")
            
            if not subject_id:
                return {"success": False, "message": "Subject ID is required"}
            
            # Verify subject exists and belongs to user  
            subject_db = Subject_db.objects(id=subject_id, owner=user.id).first()
            if not subject_db:
                return {"success": False, "message": "Subject not found or access denied"}
            
            # Load subject and get full data
            subject = Subject.from_db(subject_db)
            if not subject:
                return {"success": False, "message": "Error loading subject"}
            
            full_data = await subject.get_full_data()
            return {"success": True, "subject_data": full_data, "message": f"Retrieved full data for subject: {subject.name}"}
        
        elif function_name == "remove_ai_accessible_subject":
            # Remove a subject from AI accessible list
            subject_id = arguments.get("subject_id", "")
            
            if not subject_id:
                return {"success": False, "message": "Subject ID is required"}
            
            ai_list = user.settings.get("ai_accessible", [])
            if subject_id not in ai_list:
                return {"success": False, "message": "Subject not in AI-accessible list"}
            
            ai_list.remove(subject_id)
            user.settings["ai_accessible"] = ai_list
            user.save()
            return {"success": True, "message": f"Removed subject from AI-accessible list"}
        
        elif function_name == "delete_subject":
            # Delete a subject and its components/widgets
            subject_id = arguments.get("subject_id", "")
            
            if not subject_id:
                return {"success": False, "message": "Subject ID is required"}
            
            # Verify subject exists and belongs to user
            subject_db = Subject_db.objects(id=subject_id, owner=user.id).first()
            if not subject_db:
                return {"success": False, "message": "Subject not found or access denied"}
            
            # Check if subject is deletable
            if not subject_db.is_deletable:
                return {"success": False, "message": "This subject cannot be deleted"}
            
            # Delete associated widgets and components
            for widget in subject_db.widgets:
                widget.delete()
            for comp in subject_db.components:
                comp.delete()
            
            subject_name = subject_db.name
            subject_db.delete()
            return {"success": True, "message": f"Subject '{subject_name}' and all associated components and widgets deleted successfully"}
        
        elif function_name == "delete_component":
            # Delete a component and remove it from its host subject
            component_id = arguments.get("component_id", "")
            
            if not component_id:
                return {"success": False, "message": "Component ID is required"}
            
            # Verify component exists and belongs to user
            component = Component_db.objects(id=component_id, owner=user.id).first()
            if not component:
                return {"success": False, "message": "Component not found or access denied"}
            
            # Check if component is deletable
            if not component.is_deletable:
                return {"success": False, "message": "This component cannot be deleted"}
            
            # Handle deletion of array metadata for array types
            if component.comp_type in ["Array_type", "Array_generic", "Array_of_pairs"]:
                array_result = Arrays.delete_array(
                    user_id=user.id,
                    component_id=component_id
                )
                if not array_result["success"]:
                    return {"success": False, "message": f"Failed to delete array data: {array_result['message']}"}
            
            # Remove component from its host subject
            host_subject = Subject_db.objects(id=component.host_subject.id).first()
            if host_subject:
                host_subject.components.remove(component_id)
                host_subject.save()
            
            component_name = component.name
            component.delete()
            return {"success": True, "message": f"Component '{component_name}' deleted successfully"}
        
        elif function_name == "delete_widget":
            # Delete a widget and associated todos if applicable
            widget_id = arguments.get("widget_id", "")
            
            if not widget_id:
                return {"success": False, "message": "Widget ID is required"}
            
            # Verify widget exists and belongs to user
            widget = Widget_db.objects(id=widget_id, owner=user.id).first()
            if not widget:
                return {"success": False, "message": "Widget not found or access denied"}
            
            # Check if widget is deletable
            if widget.is_deletable == "false":
                return {"success": False, "message": "This widget cannot be deleted"}
            
            # If it's a daily_todo widget, delete associated todos
            if widget.widget_type == "daily_todo":
                Todo_db.objects(widget_id=widget_id).delete()
            
            widget_name = widget.name
            widget.delete()
            return {"success": True, "message": f"Widget '{widget_name}' deleted successfully"}
        
        elif function_name == "delete_connection":
            # Delete a connection
            connection_id = arguments.get("connection_id", "")
            
            if not connection_id:
                return {"success": False, "message": "Connection ID is required"}
            
            # Verify connection exists and belongs to user
            connection = Connection_db.objects(id=connection_id, owner=user.id).first()
            if not connection:
                return {"success": False, "message": "Connection not found or access denied"}
            
            connection.delete()
            return {"success": True, "message": "Connection deleted successfully"}
        
        elif function_name == "delete_category":
            # Delete a category and set associated subjects to 'Uncategorized'
            category_name = arguments.get("name", "")
            
            if not category_name:
                return {"success": False, "message": "Category name is required"}
            
            # Verify category exists and belongs to user
            category = Category_db.objects(name=category_name, owner=user.id).first()
            if not category:
                return {"success": False, "message": "Category not found or access denied"}
            
            # Update all subjects in this category to "Uncategorized"
            subjects_in_category = Subject_db.objects(category=category_name, owner=user.id)
            for subject in subjects_in_category:
                subject.update(category="Uncategorized")
            
            category.delete()
            return {"success": True, "message": f"Category '{category_name}' deleted and all associated subjects set to 'Uncategorized'"}
        
        elif function_name == "delete_data_transfer":
            # Delete a data transfer
            transfer_id = arguments.get("transfer_id", "")
            
            if not transfer_id:
                return {"success": False, "message": "Transfer ID is required"}
            
            # Verify data transfer exists and user has access
            data_transfer = DataTransfer_db.objects(id=transfer_id).first()
            if not data_transfer:
                return {"success": False, "message": "Data transfer not found"}
            
            # Check if user owns the source or target component
            user_has_access = False
            if data_transfer.source_component and data_transfer.source_component.owner == user.id:
                user_has_access = True
            if data_transfer.target_component and data_transfer.target_component.owner == user.id:
                user_has_access = True
            
            if not user_has_access and not user.admin:
                return {"success": False, "message": "Not authorized to delete this data transfer"}
            
            data_transfer.delete()
            return {"success": True, "message": "Data transfer deleted successfully"}
        
        elif function_name == "delete_custom_template":
            # Delete a custom template
            template_id = arguments.get("template_id", "")
            
            if not template_id:
                return {"success": False, "message": "Template ID is required"}
            
            # Verify template exists and belongs to user
            template = CustomTemplate_db.objects(id=template_id, owner=user.id).first()
            if not template:
                return {"success": False, "message": "Template not found or access denied"}
            
            template_name = template.name
            template.delete()
            return {"success": True, "message": f"Template '{template_name}' deleted successfully"}
        
        elif function_name == "delete_todo":
            # Delete a todo item
            todo_id = arguments.get("todo_id", "")
            widget_id = arguments.get("widget_id", "")
            
            if not todo_id:
                return {"success": False, "message": "Todo ID is required"}
            
            # Verify todo exists and belongs to user
            todo = Todo_db.objects(id=todo_id, owner=user.id).first()
            if not todo:
                return {"success": False, "message": "Todo not found or access denied"}
            
            # If widget_id provided, verify it matches
            if widget_id and todo.widget_id != widget_id:
                return {"success": False, "message": "Todo does not belong to specified widget"}
            
            todo_text = todo.text
            todo.delete()
            return {"success": True, "message": f"Todo '{todo_text}' deleted successfully"}
        
        elif function_name == "delete_notification":
            # Delete a notification
            notification_id = arguments.get("notification_id", "")
            
            if not notification_id:
                return {"success": False, "message": "Notification ID is required"}
            
            # Verify notification exists and belongs to user
            notification = Notification_db.objects(id=notification_id, user_id=str(user.id)).first()
            if not notification:
                return {"success": False, "message": "Notification not found or access denied"}
            
            # Update notification count
            count_obj, _ = NotificationCount.objects.get_or_create(user_id=user.id)
            count_obj.count = str(int(count_obj.count) + 1)
            count_obj.save()
            
            notification.delete()
            return {"success": True, "message": "Notification deleted successfully"}
        
        elif function_name == "create_habit":
            name = arguments.get("name")
            description = arguments.get("description")
            frequency = arguments.get("frequency", "Daily")
            add_to_tracker = arguments.get("add_to_tracker", True)
            
            # Create habit subject using the habit template
            subject = Subject(
                name=name,
                owner=str(user.id),
                template="habit",
                category="Personal Development"
            )
            
            # Apply the habit template
            await subject.apply_template("habit")
            
            # Update the description if provided
            if description:
                for comp in subject.components:
                    if comp.name == "Description":
                        comp.data["item"] = description
                        break
            
            # Update frequency if provided
            for comp in subject.components:
                if comp.name == "Frequency":
                    comp.data["item"] = frequency
                    break
            
            # Save the habit subject
            subject.save_to_db()
            habit_id = str(subject.id)
            
            print(f"Created habit subject: {name} with ID: {habit_id}")
            
            # Add to habit tracker if requested
            if add_to_tracker:
                # Call the add_habit_to_tracker function logic
                add_result = await execute_function_call("add_habit_to_tracker", {"habit_id": habit_id}, user)
                if not add_result["success"]:
                    return {
                        "success": False,
                        "message": f"Habit created but failed to add to tracker: {add_result['message']}",
                        "habit_id": habit_id
                    }
            
            return {
                "success": True,
                "message": f"Habit '{name}' created successfully" + (" and added to tracker" if add_to_tracker else ""),
                "habit_id": habit_id
            }
        
        elif function_name == "add_habit_to_tracker":
            habit_id = arguments.get("habit_id")
            
            # Validate that the subject exists and has habit template
            habit_subject = Subject_db.objects(id=habit_id, owner=str(user.id)).first()
            
            if not habit_subject:
                return {"success": False, "message": "Habit subject not found"}
            
            if habit_subject.template != "habit":
                return {"success": False, "message": "Subject is not a habit template"}
            
            # Get or create habit tracker
            habit_tracker = await HabitTrackerManager.get_or_create_habit_tracker(str(user.id))
            
            # Check if habit is already in tracker
            habits_result = Arrays.get_array_by_name(
                user_id=str(user.id),
                host_id=str(habit_tracker.id),
                array_name="habits",
                host_type="subject"
            )
            
            if habits_result["success"]:
                existing_habits = [item["value"] for item in habits_result["array"]]
                if habit_id in existing_habits:
                    return {"success": False, "message": "Habit is already in tracker"}
            
            # Add habit to tracker
            append_result = Arrays.append_to_array(
                user_id=str(user.id),
                host_id=str(habit_tracker.id),
                value=habit_id,
                host_type="subject",
                array_name="habits"
            )
            
            if append_result["success"]:
                # Also add to daily status with false status
                Arrays.append_to_array(
                    user_id=str(user.id),
                    host_id=str(habit_tracker.id),
                    value={"key": habit_id, "value": False},
                    host_type="subject",
                    array_name="daily_status"
                )
                
                return {
                    "success": True,
                    "message": f"Habit '{habit_subject.name}' added to tracker successfully"
                }
            else:
                return {"success": False, "message": "Failed to add habit to tracker"}
        
        elif function_name == "remove_habit_from_tracker":
            habit_id = arguments.get("habit_id")
            
            # Get habit tracker
            habit_tracker = await HabitTrackerManager.get_or_create_habit_tracker(str(user.id))
            
            # Find and remove from habits array
            habits_result = Arrays.get_array_by_name(
                user_id=str(user.id),
                host_id=str(habit_tracker.id),
                array_name="habits",
                host_type="subject"
            )
            
            if not habits_result["success"]:
                return {"success": False, "message": "Could not access habits array"}
            
            # Find the index of the habit
            habit_index = None
            for i, item in enumerate(habits_result["array"]):
                if item["value"] == habit_id:
                    habit_index = i
                    break
            
            if habit_index is None:
                return {"success": False, "message": "Habit not found in tracker"}
            
            # Remove from habits array
            Arrays.delete_at_index(
                user_id=str(user.id),
                host_id=str(habit_tracker.id),
                index=habit_index,
                host_type="subject",
                array_name="habits"
            )
            
            # Remove from daily status
            daily_status_result = Arrays.get_array_by_name(
                user_id=str(user.id),
                host_id=str(habit_tracker.id),
                array_name="daily_status",
                host_type="subject"
            )
            
            if daily_status_result["success"]:
                for i, item in enumerate(daily_status_result["array"]):
                    if item["value"]["key"] == habit_id:
                        Arrays.delete_at_index(
                            user_id=str(user.id),
                            host_id=str(habit_tracker.id),
                            index=i,
                            host_type="subject",
                            array_name="daily_status"
                        )
                        break
            
            return {"success": True, "message": "Habit removed from tracker successfully"}
        
        elif function_name == "mark_habit_complete":
            habit_id = arguments.get("habit_id")
            completed = arguments.get("completed")
            
            result = await HabitTrackerManager.mark_habit_done(str(user.id), habit_id, completed)
            
            if result:
                status = "completed" if completed else "not completed"
                return {
                    "success": True,
                    "message": f"Habit marked as {status} for today"
                }
            else:
                return {"success": False, "message": "Failed to update habit status"}
        
        elif function_name == "get_daily_habits_status":
            date = arguments.get("date")
            
            result = await HabitTrackerManager.get_daily_habits_status(str(user.id), date)
            
            return {
                "success": True,
                "data": result,
                "message": f"Retrieved habits status for {date or 'today'}"
            }
        
        elif function_name == "get_habit_tracker_data":
            # Get habit tracker
            habit_tracker = await HabitTrackerManager.get_or_create_habit_tracker(str(user.id))
            
            # Get full data using Subject helper
            subject = Subject.from_db(habit_tracker)
            full_data = await subject.get_full_data()
            
            return {
                "success": True,
                "data": full_data,
                "message": "Habit tracker data retrieved successfully"
            }
        
        elif function_name == "get_habit_detailed_status":
            habit_id = arguments.get("habit_id")
            date = arguments.get("date")
            
            if not habit_id:
                return {"success": False, "message": "Habit ID is required"}
            
            # Use current date if not provided
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            
            try:
                result = await HabitTrackerManager.get_habit_detailed_status(str(user.id), habit_id, date)
                return {
                    "success": True,
                    "data": result,
                    "message": f"Retrieved detailed status for habit on {date}"
                }
            except Exception as e:
                return {"success": False, "message": f"Error getting habit details: {str(e)}"}
        
        elif function_name == "get_habits_count":
            try:
                count = await HabitTrackerManager.get_habits_count(str(user.id))
                return {
                    "success": True,
                    "count": count,
                    "message": f"You have {count} habits in your tracker"
                }
            except Exception as e:
                return {"success": False, "message": f"Error getting habits count: {str(e)}"}
        
        else:
            return {"success": False, "message": f"Unknown function: {function_name}"}
    
    except Exception as e:
        return {"success": False, "message": f"Error executing {function_name}: {str(e)}"}