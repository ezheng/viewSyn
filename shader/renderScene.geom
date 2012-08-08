#version 420
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 transform0;
uniform isampler2D depthTex0;
uniform sampler2D colorTex0;
uniform float step;
uniform float texSizeX;
uniform float texSizeY;

out vec4 projTextureCoord;



void main()
{
	int depthIndex[3];

	for(int i = 0; i<3; i++)
	{ 
		vec2 texCoord = gl_in[i].gl_Position.xy;
		depthIndex[i] = texture(depthTex0, texCoord).x;
				
	}

	if( abs(depthIndex[0] - depthIndex[1])<3 && abs(depthIndex[0] - depthIndex[2])<3 && abs(depthIndex[1] - depthIndex[2])<3 && depthIndex[0] != -1 && depthIndex[1] != -1 && depthIndex[2] != -1)
	{
		for( int i = 0; i<3; i++)
		{
			float depth = -1.0f + step * float(depthIndex[i]);
			gl_Position = transform0 * vec4(gl_in[i].gl_Position.xy *2.0f - 1.0f, depth, 1.0f);

			projTextureCoord = vec4(gl_in[i].gl_Position.xy, depth * 0.5f + 1.0f, 1.0f);
			EmitVertex();
		}
	}
	EndPrimitive();
}

//==================================================================================================================


/*if(depthIndex[i] == -1)
		{
			// add a step here to make up the holes. 
			// But here I also need to make the color texture available. 
			pixelProperty pProperty[4];
			int numOfTests = 5;
			pProperty[0] = findPosLeftRight(texCoord , numOfTests, 1);
			pProperty[1] = findPosLeftRight(texCoord,  numOfTests, -1);
			pProperty[2] = findPosUpDown(texCoord, numOfTests, 1);
			pProperty[3] = findPosUpDown(texCoord, numOfTests, -1);
		}*/

/*struct pixelProperty
{
	vec4 color;
	int dist;
	int planeIdx;
}

pixelProperty findPosLeftRight( vec2 centerPos, int numOfTests, int rightDir)  
{
	pixelProperty pProperty;
	pProperty.planeIdx = -1; pProperty.color = vec4(0.0f,0.0f,0.0f, 1.0f); pProperty.dist = endPos;

	for(int i = 1; i <= numOfTests; i++ )
	{
		vec2 newTexPos(texCoord.x + float(i * rightDir) * texSizeX, texCoord.y);
		int planeIdx = texture(depthTex0, newTexPos).x;
		if( planeIdx != -1)
		{
			pProperty.color = texture( colorTex0, newTexPos);
			pProperty.dist = i;
			pProperty.planeIdx = planeIdx;
			break;
		}
	}
	return pProperty;
}

pixelProperty findPosUpDown( vec2 centerPos, int numOfTests, int upDir)
{
	pixelProperty pProperty;
	pProperty.planeIdx = -1; pProperty.color = vec4(0.0f,0.0f,0.0f, 1.0f); pProperty.dist = endPos;

	for(int i = 1; i <= numOfTests; i++)
	{
		vec2 newTexPos(texCoord.x, texCoord.y + float( i * upDir) * texSizeY);
		int planeIdx = texture(depthTex, newTexPos).x;
		if( planeIdx != -1)
		{
			pProperty.color = texture( colorTex0, newTexPos);
			pProperty.dist = i;
			pProperty.planeIdx = planeIdx;
			break;
		}
	}
}*/