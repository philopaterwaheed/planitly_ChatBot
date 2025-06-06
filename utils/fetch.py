from datetime import datetime
from models import Subject_db, Connection_db, CustomTemplate_db, Category_db, User
from bson import ObjectId

def serialize_datetime(obj):
    """Recursively serialize datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

async def fetch(user_id, ai_subject_ids):
    data = {}
    
    # Fetch AI-accessible subjects (full data)
    data['ai_accessible'] = await fetch_subjects(user_id, ai_subject_ids)
    
    # Fetch connections (only undone ones)
    data['connections'] = await fetch_connections(user_id, ai_subject_ids)
    
    # Fetch templates
    data['templates'] = await fetch_templates(user_id)
    
    # Fetch categories
    data['categories'] = await fetch_categories(user_id)
    
    return data

async def fetch_subjects(user_id, ai_subject_ids):
    # Get all subjects for the user first
    all_subjects = list(Subject_db.objects(owner=user_id))
    all_subjects_dict = {str(subj.id): subj for subj in all_subjects}
    
    # Get AI-accessible subjects (full data) - only for specified IDs
    ai_subjects_full_data = []
    for subj_id in ai_subject_ids:
        if subj_id in all_subjects_dict:
            subj = all_subjects_dict[subj_id]
            if subj:
                # Get full data with components and widgets
                full_data = {
                    "id": str(subj.id),
                    "name": subj.name,
                    "category": subj.category,
                    "owner": str(subj.owner),
                    "created_at": subj.created_at.isoformat() if hasattr(subj, 'created_at') and subj.created_at else None,
                    "components": [],
                    "widgets": []
                }
                
                # Add components data - Fix: components are ReferenceField objects
                for component_ref in subj.components:
                    if component_ref:  # Check if reference exists
                        comp_data = {
                            "id": str(component_ref.id),
                            "name": component_ref.name,
                            "type": component_ref.comp_type,  # Fix: field is comp_type, not type
                            "data": component_ref.data,
                            "created_at": component_ref.created_at.isoformat() if hasattr(component_ref, 'created_at') and component_ref.created_at else None
                        }
                        full_data["components"].append(comp_data)
                
                # Add widgets data - Fix: widgets are ReferenceField objects
                for widget_ref in subj.widgets:
                    if widget_ref:  # Check if reference exists
                        widget_data = {
                            "id": str(widget_ref.id),
                            "name": widget_ref.name,
                            "type": widget_ref.widget_type,  # Fix: field is widget_type, not type
                            "data": widget_ref.data,
                            "created_at": widget_ref.created_at.isoformat() if hasattr(widget_ref, 'created_at') and widget_ref.created_at else None
                        }
                        full_data["widgets"].append(widget_data)
                
                # Serialize datetime objects in full_data
                ai_subjects_full_data.append(serialize_datetime(full_data))
    return ai_subjects_full_data

async def fetch_connections(user_id, ai_subject_ids):
    # Fetch only the 10 most recent undone connections, ordered by start_date (most recent first)
    # If start_date is None, it will be ordered last
    connections = list(Connection_db.objects(owner=user_id, done=False)
                      .order_by('-start_date', '-id')
                      .limit(10))
    
    # Process connections data - convert to dict format instead of JSON strings
    connections_data = []
    for conn in connections:
        conn_dict = {
            "id": str(conn.id),
            "source_subject": str(conn.source_subject.id) if conn.source_subject else None,
            "target_subject": str(conn.target_subject.id) if conn.target_subject else None,
            "con_type": conn.con_type,
            "data_transfers": [str(dt.id) for dt in conn.data_transfers] if conn.data_transfers else [],
            "owner": str(conn.owner) if conn.owner else None,  # Fix: owner is StringField, not ReferenceField
            "start_date": conn.start_date.isoformat() if conn.start_date else None,
            "end_date": conn.end_date.isoformat() if conn.end_date else None,
            "done": conn.done
        }
        connections_data.append(conn_dict)
    
    return connections_data

async def fetch_templates(user_id):
    # Get User object first since CustomTemplate_db.owner is a ReferenceField
    user = User.objects(id=user_id).first()
    if not user:
        return []
    
    # Fetch custom templates for the user
    templates = list(CustomTemplate_db.objects(owner=user))
    
    templates_data = []
    for template in templates:
        template_dict = {
            "id": str(template.id),
            "name": template.name,
            "description": template.description if hasattr(template, 'description') else None,
            "category": template.category if hasattr(template, 'category') else None,
            "data": template.data,
            "owner": str(template.owner.id),  # Fix: owner is ReferenceField to User
            "created_at": template.created_at.isoformat() if hasattr(template, 'created_at') and template.created_at else None
        }
        templates_data.append(serialize_datetime(template_dict))
    
    return templates_data

async def fetch_categories(user_id):
    # Get User object first since Category_db.owner is a ReferenceField
    user = User.objects(id=user_id).first()
    if not user:
        return []
    
    # Fetch categories for the user
    categories = list(Category_db.objects(owner=user))
    
    categories_data = []
    for category in categories:
        category_dict = {
            "id": str(category.id),
            "name": category.name,
            "description": category.description if hasattr(category, 'description') else None,  # Fix: description might not exist
            "owner": str(category.owner.id),  # Fix: owner is ReferenceField to User
            "created_at": category.created_at.isoformat() if hasattr(category, 'created_at') and category.created_at else None
        }
        categories_data.append(serialize_datetime(category_dict))
    
    return categories_data