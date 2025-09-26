import base64
import os
import logging
import sys
import uuid
import uvicorn
from dotenv import load_dotenv
from io import BytesIO
from typing import Optional, Any, Union, List, Tuple
from datetime import datetime

import PIL.Image
from google import genai
from google.genai import types
from fastmcp import FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mcp_server_gemini_image_generator.prompts import get_image_generation_prompt, get_image_transformation_prompt, get_translate_prompt
from mcp_server_gemini_image_generator.utils import save_image

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("mcp-server-gemini-image-generator")


# ==================== Gemini API Interaction ====================

async def call_gemini(
    contents: List[Any], 
    model: str = "gemini-2.0-flash-preview-image-generation", 
    config: Optional[types.GenerateContentConfig] = None, 
    text_only: bool = False
) -> Union[str, bytes]:
    """Call Gemini API with flexible configuration for different use cases.
    
    Args:
        contents: The content to send to Gemini. list containing text and/or images
        model: The Gemini model to use
        config: Optional configuration for the Gemini API call
        text_only: If True, extract and return only text from the response
        
    Returns:
        If text_only is True: str - The text response from Gemini
        Otherwise: bytes - The binary image data from Gemini
        
    Raises:
        Exception: If there's an error calling the Gemini API
    """
    try:
        # Load environment variables (ensure they're loaded)
        load_dotenv()
        
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        logger.info(f"API Key loaded successfully (length: {len(api_key)})")
        
        client = genai.Client(api_key=api_key)
        
        # THE FIX: Set up proper config for image generation model
        if config is None:
            if "image-generation" in model.lower():
                # For image generation models, we need both TEXT and IMAGE modalities
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                )
                logger.info("Config set for image generation: TEXT + IMAGE modalities")
            else:
                # For regular models, use default config
                config = types.GenerateContentConfig()
        
        # Generate content using Gemini
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        logger.info(f"Response received from Gemini API using model {model}")
        
        # For text-only calls, extract just the text
        if text_only:
            text_content = response.candidates[0].content.parts[0].text.strip()
            logger.info(f"Extracted text content: {len(text_content)} characters")
            return text_content
        
        # Return the image data
        logger.info("Looking for image data in response parts...")
        for i, part in enumerate(response.candidates[0].content.parts):
            if part.inline_data is not None:
                image_data = part.inline_data.data
                logger.info(f"Found image data in part {i}, size: {len(image_data)} bytes")
                return image_data
        
        raise ValueError("No image data found in Gemini response")
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise

# ==================== Text Utility Functions ====================

async def convert_prompt_to_filename(prompt: str) -> str:
    """Convert a text prompt into a suitable filename for the generated image using Gemini AI.
    
    Args:
        prompt: The text prompt used to generate the image
        
    Returns:
        A concise, descriptive filename generated based on the prompt
    """
    try:
        # Create a prompt for Gemini to generate a filename
        filename_prompt = f"""
        Based on this image description: "{prompt}"
        
        Generate a short, descriptive file name suitable for the requested image.
        The filename should:
        - Be concise (maximum 5 words)
        - Use underscores between words
        - Not include any file extension
        - Only return the filename, nothing else
        """
        
        # Call Gemini and get the filename
        generated_filename = await call_gemini(filename_prompt, text_only=True)
        logger.info(f"Generated filename: {generated_filename}")
        
        # Return the filename only, without path or extension
        return generated_filename
    
    except Exception as e:
        logger.error(f"Error generating filename with Gemini: {str(e)}")
        # Fallback to a simple filename if Gemini fails
        truncated_text = prompt[:12].strip()
        return f"image_{truncated_text}_{str(uuid.uuid4())[:8]}"


async def translate_prompt(text: str) -> str:
    """Translate and optimize the user's prompt to English for better image generation results.
    
    Args:
        text: The original prompt in any language
        
    Returns:
        English translation of the prompt with preserved intent
    """
    try:
        # Create a prompt for translation with strict intent preservation
        prompt = get_translate_prompt(text)

        # Call Gemini and get the translated prompt
        translated_prompt = await call_gemini(prompt, text_only=True)
        logger.info(f"Original prompt: {text}")
        logger.info(f"Translated prompt: {translated_prompt}")
        
        return translated_prompt
    
    except Exception as e:
        logger.error(f"Error translating prompt: {str(e)}")
        # Return original text if translation fails
        return text

   
# ==================== Image Processing Functions ====================

async def save_image_to_path(image_bytes: bytes, filename: str) -> str:
    """Save image bytes to the configured output directory.
    
    Args:
        image_bytes: Raw image data as bytes
        filename: Name for the saved file
        
    Returns:
        Full path to the saved image file
    """
    try:
        # Get output directory from environment variable
        output_dir = os.environ.get("OUTPUT_IMAGE_PATH", "./generated_images")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
        
        # Ensure filename has proper extension
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            filename = f"{filename}.jpg"
        
        # Create full file path
        full_path = os.path.join(output_dir, filename)
        
        # Save the image
        with open(full_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.info(f"Image saved as: {full_path}")
        return os.path.abspath(full_path)
        
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise


async def process_image_with_gemini(
    contents: List[Any], 
    prompt: str, 
    model: str = "gemini-1.5-flash"
) -> Tuple[bytes, str]:
    """Process an image request with Gemini and save the result.
    
    Args:
        contents: List containing the prompt and optionally an image
        prompt: Original prompt for filename generation
        model: Gemini model to use
        
    Returns:
        Tuple of (image_bytes, saved_image_path)
    """
    # Call Gemini Vision API
    gemini_response = await call_gemini(
        contents,
        model=model,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )
    
    # Generate a filename for the image
    filename = await convert_prompt_to_filename(prompt)
    
    # Save the image and return the path
    saved_image_path = await save_image_to_path(gemini_response, filename)

    return gemini_response, saved_image_path


async def process_image_transform(
    source_image: PIL.Image.Image, 
    optimized_edit_prompt: str, 
    original_edit_prompt: str
) -> Tuple[bytes, str]:
    """Process image transformation with Gemini.
    
    Args:
        source_image: PIL Image object to transform
        optimized_edit_prompt: Optimized text prompt for transformation
        original_edit_prompt: Original user prompt for naming
        
    Returns:
        Tuple of (image_bytes, saved_image_path)
    """
    # Create prompt for image transformation
    edit_instructions = get_image_transformation_prompt(optimized_edit_prompt)
    
    # Process with Gemini and return the result
    return await process_image_with_gemini(
        [edit_instructions, source_image],
        original_edit_prompt
    )


async def load_image_from_base64(encoded_image: str) -> Tuple[PIL.Image.Image, str]:
    """Load an image from a base64-encoded string.
    
    Args:
        encoded_image: Base64 encoded image data with header
        
    Returns:
        Tuple containing the PIL Image object and the image format
    """
    if not encoded_image.startswith('data:image/'):
        raise ValueError("Invalid image format. Expected data:image/[format];base64,[data]")
    
    try:
        # Extract the base64 data from the data URL
        image_format, image_data = encoded_image.split(';base64,')
        image_format = image_format.replace('data:', '')  # Get the MIME type e.g., "image/png"
        image_bytes = base64.b64decode(image_data)
        source_image = PIL.Image.open(BytesIO(image_bytes))
        logger.info(f"Successfully loaded image with format: {image_format}")
        return source_image, image_format
    except ValueError as e:
        logger.error(f"Error: Invalid image data format: {str(e)}")
        raise ValueError("Invalid image data format. Image must be in format 'data:image/[format];base64,[data]'")
    except base64.binascii.Error as e:
        logger.error(f"Error: Invalid base64 encoding: {str(e)}")
        raise ValueError("Invalid base64 encoding. Please provide a valid base64 encoded image.")
    except PIL.UnidentifiedImageError:
        logger.error("Error: Could not identify image format")
        raise ValueError("Could not identify image format. Supported formats include PNG, JPEG, GIF, WebP.")
    except Exception as e:
        logger.error(f"Error: Could not load image: {str(e)}")
        raise


# ==================== MCP Tools (Fixed for JSON Serialization) ====================

@mcp.tool()
async def generate_image_from_text(prompt: str) -> str:
    """Generate an image based on the given text prompt using Google's Gemini model.
    Returns base64 encoded image data that can be used in HTML img tags.

    Args:
        prompt: User's text prompt describing the desired image to generate
        
    Returns:
        Base64 encoded image data as data URL (data:image/jpeg;base64,...)
    """
    try:
        logger.info(f"Generating image for prompt: {prompt}")
        0
        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
        
        # Create detailed generation prompt
        contents = get_image_generation_prompt(translated_prompt)
        
        # Get image bytes from Gemini
        image_bytes = await call_gemini(
            [contents],
            text_only=False
        )
        
        # Generate filename and save to file
        filename = await convert_prompt_to_filename(prompt)
        saved_path = await save_image_to_path(image_bytes, filename)
        logger.info(f"Image saved to: {saved_path}")
        
        # Convert to base64 for MCP response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{image_base64}"
        
        logger.info(f"Image generated successfully. Size: {len(image_bytes)} bytes")
        return data_url
        
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def caption_generator(platform: str, brand: str, prompt: str, tone: str) -> str:
    """
    Generate a short marketing caption for social media.

    Args:
        platform: Target platform (e.g., "linkedin", "twitter", "instagram")
        brand: The brand name to include or align with
        prompt: The campaign/product/event prompt
        tone: The tone of the caption (e.g., "professional", "funny", "inspiring")

    Returns:
        A single short caption as a string
    """
    try:
        # Build the structured instruction
        system_prompt = f"""
        You are a social media copywriter AI.

        Write ONE short caption for {platform}.
        Brand: {brand}
        Campaign: {prompt}
        Tone: {tone}

        Rules:
        - Must be optimized for {platform}
        - Be catchy and engaging
        - Keep it short (max 2 sentences)
        - Return ONLY the caption text, nothing else
        """

        # Call Gemini text-only
        caption = await call_gemini([system_prompt], model="gemini-1.5-flash", text_only=True)
        return caption

    except Exception as e:
        logger.error(f"Error in caption_generator: {str(e)}")
        return f"Error generating caption: {str(e)}"


@mcp.tool()
async def generate_image_complete(prompt: str, save_to_file: bool = True, filename: Optional[str] = None) -> dict:
    """Generate an image with complete response including base64 data and file path.

    Args:
        prompt: User's text prompt describing the desired image to generate
        save_to_file: Whether to save the image to file (default: True)
        filename: Optional custom filename
        
    Returns:
        Dictionary containing image data, file path, and metadata
    """
    try:
        logger.info(f"Generating complete image response for: {prompt}")
        
        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
        
        # Create detailed generation prompt
        contents = get_image_generation_prompt(translated_prompt)
        
        # Get image bytes from Gemini
        image_bytes = await call_gemini(
            [contents],
            text_only=False
        )
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{image_base64}"
        
        result = {
            "prompt": prompt,
            "translated_prompt": translated_prompt,
            "image_data": data_url,
            "size_bytes": len(image_bytes),
            "file_saved": False,
            "file_path": None
        }
        
        # Save to file if requested
        if save_to_file:
            if not filename:
                filename = await convert_prompt_to_filename(prompt)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename}_{timestamp}"
            
            saved_path = await save_image_to_path(image_bytes, filename)
            result["file_saved"] = True
            result["file_path"] = saved_path
        
        logger.info(f"Complete image generation successful")
        return result
        
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def transform_image_from_encoded(encoded_image: str, prompt: str) -> str:
    """Transform an existing image based on the given text prompt using Google's Gemini model.
    Returns base64 encoded transformed image data.

    Args:
        encoded_image: Base64 encoded image data with header. Must be in format:
                    "data:image/[format];base64,[data]"
                    Where [format] can be: png, jpeg, jpg, gif, webp, etc.
        prompt: Text prompt describing the desired transformation or modifications
        
    Returns:
        Base64 encoded transformed image data as data URL
    """
    try:
        logger.info(f"Processing transform_image_from_encoded request with prompt: {prompt}")

        # Load and validate the image
        source_image, _ = await load_image_from_base64(encoded_image)
        
        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
        
        # Process the transformation
        image_bytes, saved_path = await process_image_transform(source_image, translated_prompt, prompt)
        logger.info(f"Transformed image saved to: {saved_path}")
        
        # Convert to base64 for MCP response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{image_base64}"
        
        return data_url
        
    except Exception as e:
        error_msg = f"Error transforming image: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def transform_image_from_file(image_file_path: str, prompt: str) -> str:
    """Transform an existing image file based on the given text prompt using Google's Gemini model.
    Returns base64 encoded transformed image data.

    Args:
        image_file_path: Path to the image file to be transformed
        prompt: Text prompt describing the desired transformation or modifications
        
    Returns:
        Base64 encoded transformed image data as data URL
    """
    try:
        logger.info(f"Processing transform_image_from_file request with prompt: {prompt}")
        logger.info(f"Image file path: {image_file_path}")

        # Validate file path
        if not os.path.exists(image_file_path):
            raise ValueError(f"Image file not found: {image_file_path}")

        # Translate the prompt to English
        translated_prompt = await translate_prompt(prompt)
            
        # Load the source image directly using PIL
        try:
            source_image = PIL.Image.open(image_file_path)
            logger.info(f"Successfully loaded image from file: {image_file_path}")
        except PIL.UnidentifiedImageError:
            logger.error("Error: Could not identify image format")
            raise ValueError("Could not identify image format. Supported formats include PNG, JPEG, GIF, WebP.")
        except Exception as e:
            logger.error(f"Error: Could not load image: {str(e)}")
            raise 
        
        # Process the transformation
        image_bytes, saved_path = await process_image_transform(source_image, translated_prompt, prompt)
        logger.info(f"Transformed image saved to: {saved_path}")
        
        # Convert to base64 for MCP response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{image_base64}"
        
        return data_url
        
    except Exception as e:
        error_msg = f"Error transforming image: {str(e)}"
        logger.error(error_msg)
        return error_msg


def main():
    logger.info("Starting Gemini Image Generator MCP server...")
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",  # Standard localhost
        port=8002          # Standard port (you can change this)
    )    
    logger.info("Server stopped")

if __name__ == "__main__":
    main()