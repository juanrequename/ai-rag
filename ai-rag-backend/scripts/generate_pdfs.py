"""
CV Generation Script using LangChain
Generates unique fake CVs in PDF format with structured output validation.
"""
import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Optional, List

import requests
from dotenv import load_dotenv
from fpdf import FPDF
from pydantic import BaseModel, EmailStr, field_validator, ValidationError, Field

load_dotenv()
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


OUTPUT_DIR = Path(__file__).parent.parent / "pdf_files"
OUTPUT_DIR.mkdir(exist_ok=True)

ROLES = [
    "Software Engineer",
    "Backend Engineer",
    "Frontend Engineer",
    "DevOps Engineer",
    "Site Reliability Engineer",
    "Data Engineer",
    "Machine Learning Engineer",
    "Security Engineer",
    "Cloud Engineer",
    "QA Engineer",
]


# Pydantic Models for CV Data Validation
class Experience(BaseModel):
    """Work experience entry."""
    title: str
    company: str
    location: str
    dates: str
    responsibilities: List[str]
    
    @field_validator('responsibilities')
    @classmethod
    def validate_responsibilities(cls, v: List[str]) -> List[str]:
        if not v or len(v) < 1:
            raise ValueError('Must have at least 1 responsibility')
        if len(v) > 10:
            raise ValueError('Too many responsibilities (max 10)')
        return v


class Education(BaseModel):
    """Education entry."""
    degree: str
    institution: str
    location: str
    year: str


class CVData(BaseModel):
    """Complete CV data structure with validation."""
    full_name: str = Field(
        description="Full name in format 'First Last' (e.g., 'John Smith'). Use real names, not fake ones.",
        examples=["John Smith", "Carlos Rodriguez"]
    )
    email: EmailStr = Field(
        description="Professional email address using the person's name",
        examples=["john.smith@gmail.com"]
    )
    phone: str = Field(
        description="Phone number in international format",
        examples=["+1 (555) 123-4567", "+44 20 7946 0958"]
    )
    location: str = Field(
        description="City, Country format",
        examples=["San Francisco, USA", "London, UK", "Berlin, Germany"]
    )
    linkedin: str = Field(
        description="LinkedIn profile URL",
        examples=["linkedin.com/in/john-smith"]
    )
    summary: str = Field(
        description="2-3 sentence professional summary highlighting key strengths",
        min_length=50,
        max_length=500
    )
    skills: List[str] = Field(
        description="Technical skills relevant to the role",
        min_length=4,
        max_length=20,
        examples=["Python", "SQL", "AWS", "Docker", "Git", "React", "Node.js", "REST APIs"]
    )
    experience: List[Experience]
    education: List[Education]
    languages: List[str]
    
    @field_validator('full_name')
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError('Full name must be at least 2 characters')
        return v.strip()
    
    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v: List[str]) -> List[str]:
        if len(v) < 4:
            raise ValueError('Must have at least 4 skills')
        if len(v) > 20:
            raise ValueError('Too many skills (max 20)')
        return v
    
    @field_validator('experience')
    @classmethod
    def validate_experience(cls, v: List[Experience]) -> List[Experience]:
        if len(v) < 1:
            raise ValueError('Must have at least 1 work experience')
        if len(v) > 10:
            raise ValueError('Too many work experiences (max 10)')
        return v
    
    @field_validator('education')
    @classmethod
    def validate_education(cls, v: List[Education]) -> List[Education]:
        if len(v) < 1:
            raise ValueError('Must have at least 1 education entry')
        return v
    
    @field_validator('languages')
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError('Must have at least 1 language')
        return v


cv_prompt = PromptTemplate(
    input_variables=["role"],
    template="""Generate a fake but realistic CV for a person applying for a {role} role.

Requirements:
- Make all information fictional but realistic
- Include 2-3 work experiences with 3-4 responsibilities each
- Include 4-8 relevant technical skills for the role
- Include 1-2 education entries
- Use diverse names (e.g. Michael, James, John, Louis, Noah, etc), backgrounds, and locations
- Make the professional summary natural and compelling (2-3 sentences)
- Include at least 1 language

The CV will be automatically structured according to the schema."""
)


def generate_cv_data(role: str, cv_chain) -> CVData:
    """Generate CV data using LangChain structured output.
    
    Args:
        role: The job role to generate CV for
        cv_chain: LangChain chain with structured output configured
        
    Returns:
        Validated CVData object
        
    Raises:
        ValidationError: If LLM output doesn't match schema
    """
    cv_data = cv_chain.invoke({"role": role})
    return cv_data


def generate_cv_image(role: str, image_generator: DallEAPIWrapper) -> Optional[str]:
    """Generate a professional avatar/headshot image for a CV.
    
    Args:
        role: The job role for context in generating an appropriate image.
        image_generator: DallEAPIWrapper instance for image generation.
    
    Returns:
        Path to the downloaded image file, or None if generation/download failed.
    """
    prompt = f"Professional headshot portrait photo of a {role}, corporate style, neutral background, high quality, photorealistic"
    image_url = image_generator.run(prompt)
    
    # Download image to temp file
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        suffix = ".png"
        if "jpeg" in response.headers.get("content-type", ""):
            suffix = ".jpg"
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except requests.RequestException as err:
        print(f"Failed to download image: {err}")
        return None


def create_pdf(cv_data: CVData, image_path: Optional[str] = None) -> str:
    """Create a PDF from validated CV data.
    
    Args:
        cv_data: Validated CVData object
        image_path: Optional path to profile image
        
    Returns:
        Path to the created PDF file
    """

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    header_color = (30, 58, 95)
    accent_color = (37, 99, 235)
    text_color = (31, 41, 55)
    muted = (107, 114, 128)

    # Profile image dimensions
    img_size = 35
    img_x = 165
    img_y = 10

    # Add profile image if provided
    if image_path and os.path.exists(image_path):
        pdf.image(image_path, x=img_x, y=img_y, w=img_size, h=img_size)

    pdf.set_text_color(*header_color)
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 12, cv_data.full_name, ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*muted)
    contact_line = f"{cv_data.email}  |  {cv_data.phone}  |  {cv_data.location}"
    pdf.cell(0, 6, contact_line, ln=True)
    if cv_data.linkedin:
        pdf.cell(0, 6, cv_data.linkedin, ln=True)

    # Ensure we clear past the image area
    if image_path and os.path.exists(image_path):
        min_y = img_y + img_size + 4
        if pdf.get_y() < min_y:
            pdf.set_y(min_y)

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 8, "PROFESSIONAL SUMMARY", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*text_color)
    pdf.multi_cell(0, 5, cv_data.summary)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 8, "WORK EXPERIENCE", ln=True)
    for exp in cv_data.experience:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*text_color)
        pdf.cell(0, 6, exp.title, ln=True)

        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*muted)
        company_line = f"{exp.company} | {exp.location} | {exp.dates}"
        pdf.cell(0, 5, company_line, ln=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*text_color)
        for resp in exp.responsibilities:
            pdf.multi_cell(0, 5, f"  * {resp}")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 8, "SKILLS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*text_color)
    pdf.multi_cell(0, 5, " | ".join(cv_data.skills))
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*accent_color)
    pdf.cell(0, 8, "EDUCATION", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*text_color)
    for edu in cv_data.education:
        pdf.cell(0, 6, edu.degree, ln=True)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*muted)
        edu_line = f"{edu.institution} | {edu.location} | {edu.year}"
        pdf.cell(0, 5, edu_line, ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*text_color)

    if cv_data.languages:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*accent_color)
        pdf.cell(0, 8, "LANGUAGES", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*text_color)
        pdf.multi_cell(0, 5, " | ".join(cv_data.languages))

    safe_name = cv_data.full_name.lower().replace(" ", "_")
    output_path = OUTPUT_DIR / f"cv_{safe_name}.pdf"
    counter = 1
    while output_path.exists():
        output_path = OUTPUT_DIR / f"cv_{safe_name}_{counter}.pdf"
        counter += 1

    pdf.output(str(output_path))
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake CVs in PDF format")
    parser.add_argument(
        "-n",
        type=int,
        default=25,
        help="Number of CVs to generate (default: 25)"
    )
    args = parser.parse_args()
    
    api_key = os.getenv("RAG__OPENAI_API_KEY")
    if not api_key:
        print("Set RAG__OPENAI_API_KEY and rerun.")
        return

    # Configure LLM with structured output for strict validation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=api_key)
    structured_llm = llm.with_structured_output(CVData, method="json_schema", strict=True)
    cv_chain = cv_prompt | structured_llm
    
    image_generator = DallEAPIWrapper(api_key=api_key, size="256x256")

    num_cvs = args.n
    selected_roles = random.sample(ROLES, min(len(ROLES), num_cvs))
    while len(selected_roles) < num_cvs:
        selected_roles.append(random.choice(ROLES))

    print(f"Generating {num_cvs} CVs into {OUTPUT_DIR}")
    for idx, role in enumerate(selected_roles, start=1):
        print(f"[{idx}/{num_cvs}] role={role}")
        image_path: Optional[str] = None
        
        try:
            # Generate CV data with structured output (automatically validated)
            cv_data = generate_cv_data(role, cv_chain)
            print(f"✓ Generated CV for: {cv_data.full_name}")
            
            # Generate profile image
            print(f"  Generating profile image...")
            image_path = generate_cv_image(role, image_generator)
            
            # Create PDF with validated data
            pdf_path = create_pdf(cv_data, image_path)
            print(f"✓ Saved to {pdf_path}\n")
            
        except ValidationError as e:
            print(f"✗ Validation failed for {role}: {e}")
            print("Skipping this CV and continuing...\n")
            continue
        except Exception as e:
            print(f"✗ Error generating CV for {role}: {e}")
            print("Skipping this CV and continuing...\n")
            continue
        finally:
            # Clean up temp image file
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)


if __name__ == "__main__":
    main()
