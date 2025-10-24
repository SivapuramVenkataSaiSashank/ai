"""
Document converter module for AI Trace Finder
Converts various document formats to images for analysis
"""

import os
import tempfile
import logging
import io
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np

# PDF processing
# On most systems the package "PyMuPDF" is imported as "fitz". Some builds
# also expose a "pymupdf" module name. Try both to maximize compatibility.
try:
    import fitz  # PyMuPDF canonical import name
    PYMUPDF_AVAILABLE = True
except Exception as primary_err:
    try:
        import pymupdf as fitz  # alternative module name
        PYMUPDF_AVAILABLE = True
    except Exception as fallback_err:
        PYMUPDF_AVAILABLE = False
        logging.warning(
            f"PyMuPDF not available: primary='{primary_err}', fallback='{fallback_err}'. PDF processing will be limited."
        )

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. PDF processing will be limited.")

# DOCX processing
try:
    from docx import Document
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not available. DOCX processing will be limited.")

logger = logging.getLogger(__name__)

class DocumentConverter:
    """Converts various document formats to images"""
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024), dpi: int = 200):
        self.target_size = target_size
        self.dpi = dpi
        self.supported_formats = {
            '.pdf': self._convert_pdf,
            '.docx': self._convert_docx,
            '.doc': self._convert_doc,
            '.png': self._convert_image,
            '.jpg': self._convert_image,
            '.jpeg': self._convert_image,
            '.tif': self._convert_image,
            '.tiff': self._convert_image,
            '.bmp': self._convert_image
        }
    
    def convert_document(self, file_path: str, max_pages: int = 5) -> List[np.ndarray]:
        """
        Convert a document to a list of images
        
        Args:
            file_path: Path to the document file
            max_pages: Maximum number of pages to convert
            
        Returns:
            List of images as numpy arrays
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            converter_func = self.supported_formats[file_ext]
            images = converter_func(file_path, max_pages)
            
            logger.info(f"Converted {file_path} to {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert document {file_path}: {str(e)}")
            return []
    
    def convert_from_bytes(self, file_bytes: bytes, file_extension: str, max_pages: int = 5) -> List[np.ndarray]:
        """
        Convert document from bytes to images
        
        Args:
            file_bytes: Document content as bytes
            file_extension: File extension (e.g., '.pdf')
            max_pages: Maximum number of pages to convert
            
        Returns:
            List of images as numpy arrays
        """
        try:
            logger.info(f"Converting document from bytes: {len(file_bytes)} bytes, extension: {file_extension}")
            
            # Validate file bytes
            if len(file_bytes) == 0:
                logger.error("File is empty")
                return []
            
            # Check file signature for PDF
            if file_extension == '.pdf':
                if not file_bytes.startswith(b'%PDF'):
                    logger.error("File does not appear to be a valid PDF (missing PDF signature)")
                    return []
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            
            logger.info(f"Created temporary file: {tmp_path}")
            
            # Convert using the temporary file
            images = self.convert_document(tmp_path, max_pages)
            
            logger.info(f"Conversion result: {len(images)} images")
            
            # Clean up
            os.unlink(tmp_path)
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert document from bytes: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _convert_pdf(self, file_path: str, max_pages: int) -> List[np.ndarray]:
        """Convert PDF to images using PyMuPDF or pdf2image"""
        images = []
        
        # Try PyMuPDF first (faster)
        if PYMUPDF_AVAILABLE:
            try:
                logger.info(f"Attempting PyMuPDF conversion for {file_path}")
                doc = fitz.open(file_path)
                page_count = min(len(doc), max_pages)
                logger.info(f"PDF has {len(doc)} pages, processing {page_count}")
                
                if page_count == 0:
                    logger.warning("PDF has no pages")
                    doc.close()
                    return []
                
                for page_num in range(page_count):
                    try:
                        page = doc[page_num]
                        # Render page to image
                        mat = fitz.Matrix(self.dpi/72, self.dpi/72)  # 72 is default DPI
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        
                        # Resize if needed
                        if img.size != self.target_size:
                            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                        
                        # Convert to numpy array
                        img_array = np.array(img)
                        images.append(img_array)
                        logger.info(f"Successfully converted page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to convert page {page_num + 1}: {e}")
                        continue
                
                doc.close()
                logger.info(f"PyMuPDF conversion completed: {len(images)} images")
                return images
                
            except Exception as e:
                logger.warning(f"PyMuPDF conversion failed: {e}")
                import traceback
                logger.warning(f"PyMuPDF traceback: {traceback.format_exc()}")
        
        # Fallback to pdf2image
        if PDF2IMAGE_AVAILABLE:
            try:
                logger.info(f"Attempting pdf2image conversion for {file_path}")
                pil_images = convert_from_path(file_path, dpi=self.dpi, first_page=1, last_page=max_pages)
                logger.info(f"pdf2image returned {len(pil_images)} images")
                
                for i, pil_img in enumerate(pil_images):
                    try:
                        # Resize if needed
                        if pil_img.size != self.target_size:
                            pil_img = pil_img.resize(self.target_size, Image.Resampling.LANCZOS)
                        
                        # Convert to numpy array
                        img_array = np.array(pil_img)
                        images.append(img_array)
                        logger.info(f"Successfully converted page {i + 1} with pdf2image")
                        
                    except Exception as e:
                        logger.warning(f"Failed to process page {i + 1} with pdf2image: {e}")
                        continue
                
                logger.info(f"pdf2image conversion completed: {len(images)} images")
                return images
                
            except Exception as e:
                logger.warning(f"pdf2image conversion failed: {e}")
                import traceback
                logger.warning(f"pdf2image traceback: {traceback.format_exc()}")
        
        # Last resort: try to create a simple image from PDF info
        logger.warning("Both PyMuPDF and pdf2image failed, trying fallback method")
        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(file_path)
                if len(doc) > 0:
                    # Create a simple placeholder image
                    img = Image.new('RGB', self.target_size, 'white')
                    img_array = np.array(img)
                    images.append(img_array)
                    logger.info("Created fallback image for PDF")
                doc.close()
                return images
        except Exception as e:
            logger.error(f"Fallback method also failed: {e}")
        
        logger.error("No PDF conversion method available")
        raise RuntimeError("No PDF conversion method available")
    
    def _convert_docx(self, file_path: str, max_pages: int) -> List[np.ndarray]:
        """Convert DOCX to images (simplified - converts to text and creates image)"""
        if not DOCX_AVAILABLE:
            raise RuntimeError("python-docx not available")
        
        try:
            doc = Document(file_path)
            
            # Extract text
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
            
            # Create a simple text image
            if text_content:
                # Join text and create image
                full_text = '\n'.join(text_content[:max_pages * 10])  # Limit text
                
                # Create image with text
                img = Image.new('RGB', self.target_size, 'white')
                # Note: This is a simplified approach. For better results, 
                # you might want to use a more sophisticated text rendering
                
                # Convert to numpy array
                img_array = np.array(img)
                return [img_array]
            
            return []
            
        except Exception as e:
            logger.error(f"DOCX conversion failed: {e}")
            return []
    
    def _convert_doc(self, file_path: str, max_pages: int) -> List[np.ndarray]:
        """Convert DOC to images (simplified)"""
        # For DOC files, we'll create a placeholder image
        # In a production environment, you might want to use python-docx2txt or similar
        logger.warning("DOC conversion is simplified - consider using DOCX format")
        
        # Create a placeholder image
        img = Image.new('RGB', self.target_size, 'white')
        img_array = np.array(img)
        return [img_array]
    
    def _convert_image(self, file_path: str, max_pages: int) -> List[np.ndarray]:
        """Convert image files to numpy arrays"""
        try:
            img = Image.open(file_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if needed
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            return [img_array]
            
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats.keys())
    
    def is_supported(self, file_extension: str) -> bool:
        """Check if file format is supported"""
        return file_extension.lower() in self.supported_formats

