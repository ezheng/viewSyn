#version 420

layout(location = 0) out vec4 fragmentColor;
in vec2 texCoordOut;

uniform sampler3D texs;
uniform int numOfLayers;



void main()
{
	
	fragmentColor = vec4(0,0,0,1.0);

	float size = 0.2;	// determine number of textures per row
	
	float col = floor(texCoordOut.x / size);
	float row = floor(texCoordOut.y / size);

	float xOffset = texCoordOut.x - col * size;
	float yOffset = texCoordOut.y - row * size;

	float i = row * 5 + col;
	float layer = (1.0f +  2.0f *  i) / (2* float(numOfLayers));

	fragmentColor.xyz = fragmentColor.xyz + texture(texs, vec3(xOffset/size, yOffset/size , layer)).xyz;
	//if(fragmentColor.x == 0 && fragmentColor.y == 0 && fragmentColor.z == 0)
	//	fragmentColor = vec4(1.0,1.0f,1.0f, 1.0f);

}