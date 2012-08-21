#version 420
in vec4 ProjTexCoord[2];
in vec2 tex2dCoord;
uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

layout(location = 0) out float meanCost;

void main()
{
	vec4 projTexColor[2];
	bool t[2];
	for(int i = 0; i<2; i++)
	{
		t[i] = false;
	}
	
	int numOfViews = 0;
	//vec4 normalizedCoord[2];
	//normalizedCoord[0] = ProjTexCoord[0]/ProjTexCoord[0].w;

	if(ProjTexCoord[0].x/ProjTexCoord[0].w > 0 && ProjTexCoord[0].x/ProjTexCoord[0].w <1.0 &&
		 ProjTexCoord[0].y/ProjTexCoord[0].w > 0 && ProjTexCoord[0].y/ProjTexCoord[0].w < 1.0 && 
			ProjTexCoord[0].z/ProjTexCoord[0].w > 0.0f)
	/*if( normalizedCoord[0].x > 0.0 && normalizedCoord[0].x < 1.0 && 
		normalizedCoord[0].y > 0.0 && normalizedCoord[0].y < 1.0 &&
		normalizedCoord[0].z > 0.0f)*/
	{
		projTexColor[0] = textureProj(tex0, ProjTexCoord[0]);
		numOfViews = numOfViews + 1;
		t[0] = true;
	}
	if(ProjTexCoord[1].x/ProjTexCoord[1].w > 0 && ProjTexCoord[1].x/ProjTexCoord[1].w <1.0 && 
		ProjTexCoord[1].y/ProjTexCoord[1].w > 0 && ProjTexCoord[1].y/ProjTexCoord[1].w < 1.0 && 
			ProjTexCoord[1].z/ProjTexCoord[0].w > 0.0f)
	{
		projTexColor[1] = textureProj(tex1, ProjTexCoord[1]);
		numOfViews = numOfViews + 1;
		t[1] = true;
	}
	
	vec4 refColor = texture(tex2, tex2dCoord);

	meanCost = 1000000.0f;
	if(numOfViews < 1){
		meanCost = 1000000.0f;
	}
	else
	{
		for(int i = 0; i<projTexColor.length(); i++)
		{
			if(t[i] == true)
			{
				vec4 colorDiff = abs(projTexColor[i] - refColor);
				meanCost = min(meanCost, min( (colorDiff.x + colorDiff.y + colorDiff.z)/3.0f, 0.1));
			}
		}
	}
}

