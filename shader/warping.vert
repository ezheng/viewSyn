#version 420

layout(location = 0) in vec3 vertexPos;
out int instanceID;

void main()
{
	gl_Position = vec4(vertexPos, 1.0f);
	instanceID = gl_InstanceID;
}