#version 420

layout (location = 0)in vec3 vertexPosition;
layout (location = 2)in vec2 texCoord;

uniform mat4 modelViewMatrix1;
uniform mat4 projectionMatrix1;
out vec2 texCoordOut;

void main()
{
	gl_Position = projectionMatrix1 * modelViewMatrix1 * vec4(vertexPosition,1.0f);
	texCoordOut = texCoord;
}

