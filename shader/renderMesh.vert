#version 420

layout(location = 0) in vec4 vertexPos;

uniform mat4 viewpointMVP;
uniform mat4 textureMVP;
uniform mat4 textureMVP1;

out vec4 projTexCoord;
out vec4 projTexCoord1;

void main()
{
	gl_Position = viewpointMVP * vertexPos;	
	projTexCoord = textureMVP * vertexPos;
	projTexCoord1 = textureMVP1 * vertexPos;
}