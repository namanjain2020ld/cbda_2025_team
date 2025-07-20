import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import requests
from io import StringIO
import itertools
from collections import defaultdict, Counter
import random

class StudentGroupingSystem:
    def __init__(self, course_penalty=0.5, group_size=4):
        """
        Initialize the grouping system
        
        Args:
            course_penalty (float): Penalty to subtract from similarity if same course
            group_size (int): Target size for each group
        """
        self.course_penalty = course_penalty
        self.group_size = group_size
        self.data = None
        self.similarity_matrix = None
        self.groups = []
        
    def fetch_google_sheet_csv(self, sheet_url):
        """
        Fetch data from a published Google Sheet in CSV format
        
        Args:
            sheet_url (str): URL of the published Google Sheet
            
        Returns:
            pandas.DataFrame: The fetched data
        """
        try:
            # Convert Google Sheets URL to CSV export URL if needed
            if '/edit' in sheet_url:
                sheet_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
                sheet_url = sheet_url.replace('/edit', '/export?format=csv')
            elif 'docs.google.com' in sheet_url and 'export' not in sheet_url:
                sheet_url = sheet_url + '/export?format=csv'
            
            print(f"Fetching data from: {sheet_url}")
            response = requests.get(sheet_url)
            response.raise_for_status()
            
            # Read CSV data
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            print(f"Successfully loaded {len(df)} student records")
            return df
            
        except Exception as e:
            print(f"Error fetching Google Sheet: {e}")
            # Return sample data for demonstration
            # return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data that mimics a real survey format"""
        print("Using sample data for demonstration...")
        
        np.random.seed(42)
        
        # Student details (first 2 columns)
        names = [f'Student_{i+1}' for i in range(50)]
        courses = ['Computer Science', 'Mathematics', 'Physics', 'Chemistry', 'Biology'] * 10
        
        # Survey questions (remaining columns)
        sample_data = []
        for i in range(50):
            row = {
                # Student details
                'Name': names[i],
                'Course': courses[i],
                
                # Survey questions (various types)
                'Preferred_Learning_Style': np.random.choice(['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing']),
                'Study_Time_Preference': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night']),
                'Group_Work_Comfort': np.random.choice([1, 2, 3, 4, 5]),  # 1-5 scale
                'Communication_Style': np.random.choice(['Direct', 'Diplomatic', 'Casual', 'Formal']),
                'Project_Role_Preference': np.random.choice(['Leader', 'Researcher', 'Presenter', 'Organizer']),
                'Work_Pace': np.random.choice(['Fast', 'Moderate', 'Steady', 'Flexible']),
                'Conflict_Resolution': np.random.choice(['Discussion', 'Compromise', 'Voting', 'Leader_Decides']),
                'Meeting_Frequency': np.random.choice(['Daily', 'Every_2_days', 'Weekly', 'As_needed']),
                'Technology_Comfort': np.random.choice([1, 2, 3, 4, 5]),  # 1-5 scale
                'Creativity_vs_Structure': np.random.choice([1, 2, 3, 4, 5]),  # 1=Creative, 5=Structured
                'Extroversion_Level': np.random.choice([1, 2, 3, 4, 5]),  # 1-5 scale
                'Previous_Group_Experience': np.random.choice(['Excellent', 'Good', 'Average', 'Poor', 'None'])
            }
            sample_data.append(row)
        
        return pd.DataFrame(sample_data)
    
    
    def identify_columns(self, df, student_detail_cols=None):
        """
        Identify student detail columns vs survey question columns
        
        Args:
            df (pandas.DataFrame): Raw data
            student_detail_cols (list): Manual specification of student detail columns
            
        Returns:
            tuple: (detail_columns, question_columns)
        """
        columns = df.columns.tolist()
        
        if student_detail_cols:
            detail_cols = student_detail_cols
            question_cols = [col for col in columns if col not in detail_cols]
        else:
            # Auto-detect: Usually first 1-2 columns are student details
            # Look for columns that might be identifiers
            detail_cols = []
            for col in columns[:3]:  # Check first 3 columns
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['name', 'id', 'email', 'roll', 'student', 'course']):
                    detail_cols.append(col)
                elif len(detail_cols) < 2:  # Take first 2 if no clear identifiers
                    detail_cols.append(col)
            
            # If no clear identifiers found, take first 2 columns
            if not detail_cols:
                detail_cols = columns[:2]
            
            question_cols = [col for col in columns if col not in detail_cols]
        
        print(f"Student detail columns: {detail_cols}")
        print(f"Survey question columns: {question_cols}")
        
        return detail_cols, question_cols
    
    def preprocess_data(self, df, student_detail_cols=None):
        """
        Preprocess the data focusing on survey questions for similarity
        
        Args:
            df (pandas.DataFrame): Raw student data
            student_detail_cols (list): Manual specification of student detail columns
            
        Returns:
            pandas.DataFrame: Processed feature matrix for similarity computation
        """
        self.data = df.copy()
        
        # Identify which columns are student details vs survey questions
        self.detail_cols, self.question_cols = self.identify_columns(df, student_detail_cols)
        
        print(f"\nProcessing {len(self.question_cols)} survey questions for similarity matching...")
        
        # Extract only survey question columns for feature encoding
        question_data = df[self.question_cols].copy()
        
        # Clean and prepare the data
        feature_df = pd.DataFrame(index=df.index)
        
        # Process each survey question
        for col in self.question_cols:
            print(f"Processing question: {col}")
            
            # Clean the responses (remove extra spaces, handle missing values)
            question_data[col] = question_data[col].astype(str).str.strip()
            question_data[col] = question_data[col].replace(['nan', 'NaN', '', 'None'], 'Not Answered')
            
            # Check if responses are numerical or categorical
            unique_vals = question_data[col].unique()
            print(f"  Unique responses: {unique_vals[:10]}...")  # Show first 10
            
            # Try to convert to numeric if possible (for Likert scales, ratings, etc.)
            try:
                numeric_series = pd.to_numeric(question_data[col], errors='coerce')
                if not numeric_series.isna().all() and len(unique_vals) <= 10:  # Likely a scale
                    print(f"  → Treating as numeric scale")
                    # Normalize numeric responses
                    if numeric_series.std() > 0:
                        feature_df[col] = (numeric_series - numeric_series.mean()) / numeric_series.std()
                    else:
                        feature_df[col] = numeric_series
                else:
                    raise ValueError("Not numeric")
                    
            except (ValueError, TypeError):
                # Treat as categorical - use one-hot encoding
                print(f"  → Treating as categorical (one-hot encoding)")
                dummies = pd.get_dummies(question_data[col], prefix=f"Q_{col[:20]}")  # Limit prefix length
                feature_df = pd.concat([feature_df, dummies], axis=1)
        
        # Handle any NaN values
        feature_df = feature_df.fillna(0)
        
        self.feature_matrix = feature_df.values
        self.feature_columns = feature_df.columns.tolist()
        
        print(f"\nFeature matrix created:")
        print(f"  - Shape: {self.feature_matrix.shape}")
        print(f"  - Features: {len(self.feature_columns)} total features")
        print(f"  - Sample feature names: {self.feature_columns[:5]}...")
        
        return feature_df
    
    def compute_similarity_matrix(self, course_col_name=None):
        """
        Compute similarity matrix with course penalty
        
        Args:
            course_col_name (str): Name of the course column for penalty application
        """
        # Calculate basic cosine similarity based on survey responses
        base_similarity = cosine_similarity(self.feature_matrix)
        
        # Apply course penalty if course column is identified
        n_students = len(self.data)
        adjusted_similarity = base_similarity.copy()
        
        # Try to find course column automatically if not specified
        if course_col_name is None:
            # Look for course-related column in detail columns
            for col in self.detail_cols:
                if 'course' in col.lower() or 'department' in col.lower() or 'program' in col.lower():
                    course_col_name = col
                    break
        
        if course_col_name and course_col_name in self.data.columns:
            print(f"Applying course penalty using column: {course_col_name}")
            
            for i in range(n_students):
                for j in range(n_students):
                    if i != j:
                        course_i = str(self.data.iloc[i][course_col_name]).strip()
                        course_j = str(self.data.iloc[j][course_col_name]).strip()
                        
                        if course_i == course_j and course_i != 'nan':
                            adjusted_similarity[i][j] -= self.course_penalty
            
            print(f"Course penalty ({self.course_penalty}) applied for same-course pairs")
        else:
            print("No course column found - grouping based on survey similarity only")
        
        self.similarity_matrix = adjusted_similarity
        self.course_column = course_col_name
        return adjusted_similarity
    
    def greedy_grouping_algorithm(self):
        """
        Custom greedy algorithm for grouping students based on survey responses
        """
        n_students = len(self.data)
        ungrouped = list(range(n_students))
        groups = []
        
        print(f"Starting grouping for {n_students} students...")
        
        while len(ungrouped) >= self.group_size:
            group = []
            
            # Pick a random seed student
            seed_idx = random.choice(ungrouped)
            group.append(seed_idx)
            ungrouped.remove(seed_idx)
            
            # Find similar students, preferring different courses if course info available
            candidates = []
            for candidate_idx in ungrouped:
                similarity_score = self.similarity_matrix[seed_idx][candidate_idx]
                candidates.append((candidate_idx, similarity_score))
            
            # Sort by similarity (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Add students to group
            for candidate_idx, sim_score in candidates:
                if len(group) >= self.group_size:
                    break
                
                # If course information is available, check course diversity
                if hasattr(self, 'course_column') and self.course_column:
                    group_courses = [str(self.data.iloc[i][self.course_column]).strip() 
                                   for i in group if str(self.data.iloc[i][self.course_column]).strip() != 'nan']
                    candidate_course = str(self.data.iloc[candidate_idx][self.course_column]).strip()
                    
                    # Prefer different courses, but don't be too strict
                    if candidate_course not in group_courses or len(ungrouped) <= self.group_size - len(group):
                        group.append(candidate_idx)
                        ungrouped.remove(candidate_idx)
                else:
                    # No course info - just add based on similarity
                    group.append(candidate_idx)
                    ungrouped.remove(candidate_idx)
            
            groups.append(group)
            print(f"Formed group {len(groups)} with {len(group)} students")
        
        # Handle remaining students
        if ungrouped:
            if len(groups) > 0:
                print(f"Distributing {len(ungrouped)} remaining students to existing groups...")
                # Distribute remaining students to existing groups
                for student_idx in ungrouped:
                    # Find the group with highest average similarity
                    best_group_idx = 0
                    best_avg_similarity = -1
                    
                    for group_idx, group in enumerate(groups):
                        avg_sim = np.mean([self.similarity_matrix[student_idx][member] for member in group])
                        if avg_sim > best_avg_similarity:
                            best_avg_similarity = avg_sim
                            best_group_idx = group_idx
                    
                    groups[best_group_idx].append(student_idx)
            else:
                # Create a group with remaining students
                groups.append(ungrouped)
        
        self.groups = groups
        print(f"Grouping complete: {len(groups)} groups formed")
        return groups
    
    def analyze_groups(self):
        """
        Analyze the quality of the formed groups
        """
        analysis = {
            'total_groups': len(self.groups),
            'group_sizes': [len(group) for group in self.groups],
            'course_diversity': [],
            'average_similarity': [],
            'question_diversity': []
        }
        
        for i, group in enumerate(self.groups):
            # Course diversity (if course column exists)
            if hasattr(self, 'course_column') and self.course_column:
                courses_in_group = [str(self.data.iloc[idx][self.course_column]).strip() 
                                  for idx in group if str(self.data.iloc[idx][self.course_column]).strip() != 'nan']
                unique_courses = len(set(courses_in_group))
                analysis['course_diversity'].append(unique_courses)
            else:
                analysis['course_diversity'].append(1)  # No course info available
            
            # Average similarity within group (based on survey responses)
            if len(group) > 1:
                similarities = []
                for j in range(len(group)):
                    for k in range(j+1, len(group)):
                        similarities.append(self.similarity_matrix[group[j]][group[k]])
                analysis['average_similarity'].append(np.mean(similarities))
            else:
                analysis['average_similarity'].append(0)
            
            # Question response diversity
            group_responses = self.data.iloc[group][self.question_cols]
            diversity_scores = []
            for col in self.question_cols:
                unique_responses = len(set(group_responses[col].astype(str)))
                diversity_scores.append(unique_responses)
            analysis['question_diversity'].append(np.mean(diversity_scores))
        
        return analysis
    
    def export_to_excel(self, filename='student_groups.xlsx'):
        """
        Export the grouping results to Excel
        """
        # Create a results DataFrame
        results = []
        
        for group_num, group in enumerate(self.groups, 1):
            for student_idx in group:
                student_data = self.data.iloc[student_idx].to_dict()
                student_data['Group'] = f'Group_{group_num}'
                student_data['Group_Number'] = group_num
                results.append(student_data)
        
        results_df = pd.DataFrame(results)
        
        # Sort by group number
        results_df = results_df.sort_values('Group_Number')
        
        # Save to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Student_Groups', index=False)
            
            # Group analysis
            analysis = self.analyze_groups()
            analysis_df = pd.DataFrame({
                'Group': [f'Group_{i+1}' for i in range(len(self.groups))],
                'Group_Size': analysis['group_sizes'],
                'Course_Diversity': analysis['course_diversity'],
                'Avg_Similarity': analysis['average_similarity']
            })
            analysis_df.to_excel(writer, sheet_name='Group_Analysis', index=False)
            
            # Course distribution per group
            course_dist = []
            for i, group in enumerate(self.groups):
                courses = [self.data.iloc[idx]['Course'] for idx in group]
                course_count = Counter(courses)
                for course, count in course_count.items():
                    course_dist.append({
                        'Group': f'Group_{i+1}',
                        'Course': course,
                        'Count': count
                    })
            
            course_dist_df = pd.DataFrame(course_dist)
            course_dist_df.to_excel(writer, sheet_name='Course_Distribution', index=False)
        
        print(f"Results exported to {filename}")
        return results_df
    
    def run_complete_pipeline(self, sheet_url, output_filename='student_groups.xlsx', student_detail_cols=None, course_col_name=None):
        """
        Run the complete grouping pipeline
        
        Args:
            sheet_url (str): URL of the Google Sheet
            output_filename (str): Output Excel file name
            student_detail_cols (list): Manual specification of student detail columns
            course_col_name (str): Name of the course column for penalty application
        """
        print("=== Student Grouping Pipeline (Survey-Based) ===")
        
        # Step 1: Fetch data
        print("\n1. Fetching data from Google Sheets...")
        raw_data = self.fetch_google_sheet_csv(sheet_url)
        print(f"Columns in data: {list(raw_data.columns)}")
        
        # Step 2: Preprocess (focusing on survey questions)
        print("\n2. Preprocessing survey data...")
        processed_data = self.preprocess_data(raw_data, student_detail_cols)
        
        # Step 3: Compute similarity based on survey responses
        print("\n3. Computing similarity matrix based on survey responses...")
        self.compute_similarity_matrix(course_col_name)
        
        # Step 4: Form groups
        print("\n4. Forming groups based on survey similarity...")
        groups = self.greedy_grouping_algorithm()
        
        # Step 5: Analyze results
        print("\n5. Analyzing results...")
        analysis = self.analyze_groups()
        
        print(f"\n=== Results ===")
        print(f"Total students: {len(self.data)}")
        print(f"Total groups: {analysis['total_groups']}")
        print(f"Group sizes: {analysis['group_sizes']}")
        if hasattr(self, 'course_column') and self.course_column:
            print(f"Average course diversity per group: {np.mean(analysis['course_diversity']):.2f}")
        print(f"Average survey-based similarity per group: {np.mean(analysis['average_similarity']):.3f}")
        print(f"Average question response diversity per group: {np.mean(analysis['question_diversity']):.2f}")
        
        # Step 6: Export results
        print("\n6. Exporting results...")
        results_df = self.export_to_excel(output_filename)
        
        return results_df, analysis

# Example usage
if __name__ == "__main__":
    # Initialize the system
    grouping_system = StudentGroupingSystem(
        course_penalty=0.5,  # Penalty for same course students
        group_size=4         # Target group size
    )
    
    # Replace with your actual Google Sheets URL
    # Example: "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=0"
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSrt52SZcdUxBntObaTIK9X_DLs0bRLtz-jGuBEeU6lTRPtRUB7x6xHwK9jo1W3UrtUpiXZDC2uBZVg/pub?gid=810663990&single=true&output=csv"
    
    # Optional: Manually specify student detail columns if auto-detection doesn't work
    # student_detail_cols = ['Name', 'Course']  # Specify the first 1-2 columns
    # course_col_name = 'Course'  # Specify which column contains course information
    
    # Run the complete pipeline
    try:
        results, analysis = grouping_system.run_complete_pipeline(
            sheet_url=sheet_url,
            output_filename='student_groups.xlsx',
            # student_detail_cols=['Name', 'Course'],  # Uncomment if needed
            # course_col_name='Course'  # Uncomment if needed
        )
        
        print("\n=== Sample Results ===")
        print(results[['Name', 'Course', 'Group']].head(10))
        
        print("\n=== Survey Questions Used for Grouping ===")
        print(f"Questions: {grouping_system.question_cols}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        print("Running with sample data...")
        
        # Run with sample data
        results, analysis = grouping_system.run_complete_pipeline(
            sheet_url="https://docs.google.com/spreadsheets/d/e/2PACX-1vSrt52SZcdUxBntObaTIK9X_DLs0bRLtz-jGuBEeU6lTRPtRUB7x6xHwK9jo1W3UrtUpiXZDC2uBZVg/pub?gid=810663990&single=true&output=csv",  # Empty URL will trigger sample data
            output_filename='sample_student_groups.xlsx'
        )