"""
Simplified Formatter module for Bot5e
Enhances existing LLM output without aggressive restructuring
"""

import re

class DnDFormatter:
    """Formats D&D content by enhancing existing structure"""
    
    def format_response(self, text):
        """
        Main formatting function - enhances readability without breaking content
        
        Args:
            text: Raw text from LLM
            
        Returns:
            Enhanced markdown text
        """
        # Clean up the text first
        text = self._clean_text(text)
        
        # Detect content type and apply light formatting
        if self._is_statblock(text):
            return self._enhance_statblock(text)
        elif self._is_spell(text):
            return self._enhance_spell(text)
        else:
            return self._enhance_general(text)
    
    def _clean_text(self, text):
        """Basic cleanup"""
        # Fix common issues
        text = text.replace('-­‐‑', '-')
        text = text.replace('­‐‑', '-')
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _is_statblock(self, text):
        """Detect if text is a creature statblock"""
        indicators = ['Armor Class', 'Hit Points', 'STR', 'DEX', 'CON', 'Challenge']
        return sum(1 for indicator in indicators if indicator in text) >= 4
    
    def _is_spell(self, text):
        """Detect if text is a spell description"""
        indicators = ['Casting Time', 'Range', 'Components', 'Duration']
        return sum(1 for indicator in indicators if indicator in text) >= 3
    
    def _enhance_statblock(self, text):
        """
        Lightly enhance statblock formatting without aggressive restructuring
        """
        # Extract creature name if it's at the start
        lines = text.split('\n')
        formatted = []
        
        # Find the creature name line (usually has the size/type)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this looks like the creature name/type line
            if any(size in line for size in ['Medium', 'Small', 'Large', 'Tiny', 'Huge', 'Gargantuan']):
                # Split name from type
                parts = line.split('Medium')
                if len(parts) == 2:
                    creature_name = parts[0].strip()
                    type_info = 'Medium' + parts[1]
                    formatted.append(f"## {creature_name.upper()}\n")
                    formatted.append(f"*{type_info}*\n")
                    continue
                # Try other sizes
                for size in ['Small', 'Large', 'Tiny', 'Huge', 'Gargantuan']:
                    if size in line:
                        parts = line.split(size)
                        if len(parts) == 2:
                            creature_name = parts[0].strip()
                            type_info = size + parts[1]
                            formatted.append(f"## {creature_name.upper()}\n")
                            formatted.append(f"*{type_info}*\n")
                            break
                else:
                    formatted.append(line + '\n')
                continue
            
            # Format ability scores as a table if we find that line
            if line.startswith('STR') and 'DEX' in line and 'CON' in line:
                formatted.append('\n')
                formatted.append(self._create_ability_table(text))
                formatted.append('\n')
                # Skip the next line (the scores) as they're now in the table
                continue
            
            # Check if previous line was STR DEX CON (skip the score line)
            if i > 0 and 'STR' in lines[i-1] and re.match(r'^\d+\s*\([+\-]\d+\)', line):
                continue
            
            # Bold important attributes
            if line.startswith(('Armor Class:', 'Hit Points:', 'Speed:', 'Skills:', 
                               'Senses:', 'Languages:', 'Challenge:', 'Saving Throws:',
                               'Damage Vulnerabilities:', 'Damage Resistances:', 
                               'Damage Immunities:', 'Condition Immunities:')):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    formatted.append(f"**{parts[0]}:** {parts[1].strip()}\n")
                else:
                    formatted.append(line + '\n')
                continue
            
            # Format section headers
            if line in ['Actions', 'Reactions', 'Legendary Actions']:
                formatted.append(f"\n---\n\n### {line.upper()}\n\n")
                continue
            
            # Bold trait/action names (text before a period)
            if '. ' in line and not line.startswith(('Hit:', 'Melee', 'Ranged')):
                parts = line.split('. ', 1)
                if len(parts) == 2 and len(parts[0]) < 50:  # Reasonable trait name length
                    formatted.append(f"**{parts[0]}.** {parts[1]}\n\n")
                    continue
            
            formatted.append(line + '\n')
        
        return ''.join(formatted)
    
    def _create_ability_table(self, text):
        """Extract ability scores and create a markdown table"""
        # Find the ability scores line
        ability_line = None
        for line in text.split('\n'):
            if 'STR' in line and 'DEX' in line and 'CON' in line:
                ability_line = line
                break
        
        if not ability_line:
            return ""
        
        # Extract scores - look for pattern: number (modifier)
        scores = re.findall(r'(\d+)\s*\(([+\-]\d+)\)', text)
        
        if len(scores) != 6:
            return ""
        
        # Create table
        table = "| STR | DEX | CON | INT | WIS | CHA |\n"
        table += "|-----|-----|-----|-----|-----|-----|\n"
        table += "| " + " | ".join([f"{score} ({mod})" for score, mod in scores]) + " |\n"
        
        return table
    
    def _enhance_spell(self, text):
        """Enhance spell formatting"""
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted.append('\n')
                continue
            
            # Detect spell name (usually has "spell" or level in it)
            if 'level' in line.lower() and 'spell' in line.lower():
                spell_name = line.split('spell')[0].strip()
                formatted.append(f"## {spell_name.upper()}\n\n")
                formatted.append(f"*{line}*\n\n")
                continue
            
            # Bold spell attributes
            if line.startswith(('Casting Time:', 'Range:', 'Components:', 'Duration:')):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    formatted.append(f"**{parts[0]}:** {parts[1].strip()}\n\n")
                else:
                    formatted.append(line + '\n')
                continue
            
            formatted.append(line + '\n')
        
        return ''.join(formatted)
    
    def _enhance_general(self, text):
        """Enhance general text formatting"""
        # Just add proper paragraph breaks and bold keywords
        paragraphs = text.split('\n\n')
        formatted = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Bold important D&D terms
            for term in ['Advantage', 'Disadvantage', 'Attack Roll', 'Saving Throw', 
                        'Ability Check', 'Proficiency Bonus', 'Action', 'Bonus Action', 
                        'Reaction', 'Long Rest', 'Short Rest']:
                para = re.sub(f'\\b({term})\\b', r'**\1**', para, flags=re.IGNORECASE)
            
            formatted.append(para + '\n\n')
        
        return ''.join(formatted)


# Global formatter instance
formatter = DnDFormatter()


def format_dnd_response(text):
    """
    Convenience function to format any D&D response
    
    Args:
        text: Raw text from LLM
        
    Returns:
        Enhanced markdown text
    """
    return formatter.format_response(text)