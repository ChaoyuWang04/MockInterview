import { Badge } from "@/components/ui/badge";

export default function Skills({ skills }: { skills: readonly string[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {skills.map((skill) => (
        <Badge key={skill}>{skill}</Badge>
      ))}
    </div>
  );
}
