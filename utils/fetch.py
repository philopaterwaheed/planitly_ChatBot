from datetime import datetime
from models import Subject_db, Connection_db, CustomTemplate_db, Category_db, User, Subject, Component, Widget_db
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
            subj_db = all_subjects_dict[subj_id]
            if subj_db:
                try:
                    # Use the Subject helper class to get full data
                    subject = Subject.from_db(subj_db)
                    if subject:
                        # Get the full data including components and widgets with all their data
                        full_data_result = await subject.get_full_data()
                        
                        # Extract the subject data
                        subject_data = full_data_result["subject"]
                        components_data = full_data_result["components"]
                        widgets_data = full_data_result["widgets"]
                        
                        # Format the response structure
                        full_data = {
                            "id": str(subject_data["id"]),
                            "name": subject_data["name"],
                            "category": subject_data.get("category", "Uncategorized"),
                            "owner": str(subject_data["owner"]),
                            "template": subject_data.get("template", ""),
                            "is_deletable": subject_data.get("is_deletable", True),
                            "created_at": subject_data.get("created_at"),
                            "components": components_data,  # Full component data with all properties
                            "widgets": widgets_data  # Full widget data with all properties
                        }
                        
                        # Serialize datetime objects in full_data
                        ai_subjects_full_data.append(serialize_datetime(full_data))
                        
                except Exception as e:
                    print(f"Error fetching full data for subject {subj_id}: {e}")
                    # Fallback to basic data if full data fetch fails
                    basic_data = {
                        "id": str(subj_db.id),
                        "name": subj_db.name,
                        "category": subj_db.category,
                        "owner": str(subj_db.owner),
                        "created_at": subj_db.created_at.isoformat() if hasattr(subj_db, 'created_at') and subj_db.created_at else None,
                        "components": [],
                        "widgets": [],
                        "error": f"Could not fetch full data: {str(e)}"
                    }
                    ai_subjects_full_data.append(serialize_datetime(basic_data))
                    
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
            "owner": str(conn.owner) if conn.owner else None,
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
            "description": category.description if hasattr(category, 'description') else None,
            "owner": str(category.owner.id),  # Fix: owner is ReferenceField to User
            "created_at": category.created_at.isoformat() if hasattr(category, 'created_at') and category.created_at else None
        }
        categories_data.append(serialize_datetime(category_dict))
    
    return categories_data