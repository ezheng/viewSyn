#ifndef TEXTURE2D_H
#define TEXTURE2D_H


#include <Windows.h>
#include <gl/gl.h>


class texture2D
{
public:
	GLuint _textureID;

public:
	texture2D(int width, int height);
	~texture2D(void);
private:
	int _width;	// this is the width of texture
	int _height;

public:
	void create(const GLubyte *pPixels);
	void bindTexture();
	void unBindTexture();
	void upLoad(const GLubyte *pPixels, int width, int height);
	void createGL_R32UI();
	void createGL_R32I();
};

#endif