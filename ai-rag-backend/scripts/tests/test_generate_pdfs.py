from pathlib import Path

import requests

from scripts.generate_pdfs import (
    CVData,
    Education,
    Experience,
    generate_cv_image,
    infer_image_extension,
    pick_roles,
    create_pdf,
)


class _FakeResponse:
    def __init__(self, content: bytes = b"image", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": "image/png"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("bad response")


class _FakeSession:
    def get(self, url: str, timeout: int) -> _FakeResponse:
        return _FakeResponse(content=b"pngdata", status_code=200)


class _FakeImageGenerator:
    def run(self, prompt: str) -> str:
        return "https://example.com/fake.png"


def _sample_cv_data() -> CVData:
    return CVData(
        full_name="Alex Rivera",
        email="alex.rivera@example.com",
        phone="+1 555 123 4567",
        location="San Francisco, USA",
        linkedin="https://linkedin.com/in/alex-rivera",
        summary="Seasoned engineer with 10 years of experience building reliable backend services.",
        skills=["Python", "FastAPI", "SQL", "Docker"],
        experience=[
            Experience(
                title="Senior Backend Engineer",
                company="High Bar Tech",
                location="San Francisco, CA",
                dates="2018 - Present",
                responsibilities=["Owned payment platform", "Mentored junior engineers"],
            )
        ],
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="State University",
                location="California, USA",
                year="2016",
            )
        ],
        languages=["English"],
    )


def test_pick_roles_can_repeat_when_needed() -> None:
    requested = 10
    available = ["engineer", "designer", "pm"]
    roles = pick_roles(requested, available)

    assert len(roles) == requested
    assert set(roles).issubset(set(available))


def test_infer_image_extension_chooses_jpeg_and_defaults() -> None:
    assert infer_image_extension("image/jpeg") == ".jpg"
    assert infer_image_extension("IMAGE/JPG") == ".jpg"
    assert infer_image_extension("image/png") == ".png"
    assert infer_image_extension("") == ".png"


def test_generate_cv_image_downloads_using_session(tmp_path: Path) -> None:
    generator = _FakeImageGenerator()
    session = _FakeSession()

    image_path = generate_cv_image("Backend Engineer", generator, session=session)
    assert image_path is not None
    assert Path(image_path).exists()

    Path(image_path).unlink()


def test_create_pdf_writes_file(tmp_path: Path) -> None:
    cv_data = _sample_cv_data()
    pdf_path = create_pdf(cv_data, output_dir=tmp_path)

    assert Path(pdf_path).exists()

    Path(pdf_path).unlink()
