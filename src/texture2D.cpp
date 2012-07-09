#include <GL/glew.h>
#include "texture2D.h"


texture2D::texture2D(int width, int height): _width(width), _height(height)
{

}


texture2D::~texture2D(void)
{
	if(_textureID !=0)
	{
		glDeleteTextures(1, &_textureID);
	}
}

void texture2D::create(const GLubyte *pPixels)
{
	if(_textureID != 0)
		glDeleteTextures(1, &_textureID);

	glGenTextures(1, &_textureID);
	glBindTexture(GL_TEXTURE_2D, _textureID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR  );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
	// upload images
	if(pPixels == NULL)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0, GL_BGR_EXT , GL_UNSIGNED_BYTE, NULL);	
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0, GL_BGR_EXT , GL_UNSIGNED_BYTE, &pPixels[0]);	
	// unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);
}

void texture2D::upLoad(const GLubyte *pPixels, int width, int height)
{	
	if(_width == width && _height == height && _textureID != 0)
	{		 
		glBindTexture(GL_TEXTURE_2D, _textureID);
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_BGR_EXT , GL_UNSIGNED_BYTE, &pPixels[0]);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	else
	{			
		create(pPixels);
	}

}

void texture2D::bindTexture()
{
	glBindTexture(GL_TEXTURE_2D, _textureID);
}

void texture2D::unBindTexture()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}


