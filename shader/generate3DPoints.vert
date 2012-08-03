#version 420

layout(location = 0) in vec2 vertexPos;

void main()
{
	gl_Position = vec4(vertexPos, 0.0f, 1.0f);	// z will be filled with the depth in the texture
}